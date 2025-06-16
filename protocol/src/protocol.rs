//! implement debrag script: https://rubin.io/public/pdfs/delbrag-talk-btcpp-austin-2025.pdf

use std::ops::Add;
use bitcoin::blockdata::opcodes::all::*;
use bitcoin::blockdata::script::Builder;
use bitcoin::hashes::{Hash, HashEngine, sha256};
use bitcoin::script::{PushBytes, Script, ScriptBuf};
use bitcoin::secp256k1::rand::rngs::OsRng;
use bitcoin::secp256k1::{Keypair, PublicKey, Secp256k1, SecretKey};
use bitcoin::taproot::TaprootBuilderError;
use bitcoin::taproot::{LeafVersion, TapLeafHash, TaprootBuilder, TaprootSpendInfo};

use bitcoin_script_stack::optimizer;
use bitvm::hash::blake3::blake3_compute_script_with_limb;
use std::str::FromStr;

use anyhow::Result;
use bitcoin::{Amount, opcodes, Network, Transaction, Txid, Address, OutPoint, TxIn, Sequence, Witness, TxOut};
use bitcoin::absolute::LockTime;
use bitcoin::transaction::Version;
use secp256k1::{Message, XOnlyPublicKey};
use crate::utils::inner_from;

/// Helper: combine scripts (by just concatenating the raw bytes).
fn combine_scripts(fragments: &[ScriptBuf]) -> ScriptBuf {
    let mut combined = Vec::new();
    for frag in fragments {
        combined.extend(frag.to_bytes());
    }
    ScriptBuf::from_bytes(combined)
}

/// Builds a script that enforces one of two commitments to a value `Xi`
/// where the prover can reveal either preimage of H(Xi₀) or H(Xi₁).
///
/// Script logic:
/// OP_SHA256
/// OP_DUP
/// <H(Xi₀)> OP_EQUAL
/// OP_NOTIF
///     <H(Xi₁)> OP_EQUALVERIFY
/// OP_ENDIF
fn build_input_script(h_xi_0: &[u8; 32], h_xi_1: &[u8; 32]) -> ScriptBuf {
    Builder::new()
        .push_opcode(OP_SHA256)
        .push_opcode(OP_DUP)
        .push_slice(h_xi_0) // H(Xi₀)
        .push_opcode(OP_EQUAL)
        .push_opcode(OP_IF)
        .push_opcode(OP_DROP) // Always drop leftover hash
        .push_opcode(OP_ELSE)
        .push_slice(h_xi_1) // H(Xi₁)
        .push_opcode(OP_EQUALVERIFY)
        .push_opcode(OP_ENDIF)
        .into_script()
}


/// build
fn build_inputs_script(
    labels: &[([u8; 32], [u8; 32])],
    pk_musig: PublicKey,
) -> ScriptBuf {
    let commit_inputs = {
        let mut commits = Vec::new();
        for (h_xi_0, h_xi_1) in labels.iter() {
            commits.push(build_input_script(h_xi_0, h_xi_1));
        }
        let siging =
            Builder::new().push_key(&pk_musig.into()).push_opcode(OP_CHECKSIGVERIFY).into_script(); // Assumes both signatures required in leaf context
        commits.push(siging);
        combine_scripts(&commits)
    };
    commit_inputs
}

/// Build commit-timeout script. Verifier can get all the funds atfer waiting for ${t_cltv_value} blocks if the prover does not publish inputs.
fn build_commit_timeout_script(
    pk_verifier: PublicKey,
    t_cltv_value: i64,
) -> ScriptBuf {
    Builder::new()
        .push_int(t_cltv_value) // <T>
        .push_opcode(OP_CLTV)
        .push_opcode(OP_DROP)
        .push_key(&pk_verifier.into()) // <Bob>
        .push_opcode(OP_CHECKSIG)
        .into_script()
}


/// Disprove script
fn build_failgate_script(pk_musig: PublicKey, y0_hash: &[u8; 32]) -> ScriptBuf {
    Builder::new()
        .push_opcode(OP_SHA256)
        .push_slice(y0_hash)
        .push_opcode(OP_EQUALVERIFY)
        .push_key(&pk_musig.into())
        .push_opcode(OP_CHECKSIGVERIFY)
        .into_script()
}

/// Refund after CSV, no asserting or challenge happens
fn build_refund_script(pk_prover: PublicKey, csv_n: u16) -> ScriptBuf {
    Builder::new()
        .push_int(csv_n as i64)
        .push_opcode(OP_CSV)
        .push_opcode(OP_DROP)
        .push_key(&pk_prover.into())
        .push_opcode(OP_CHECKSIG)
        .into_script()
}

/// Create a minimal sighash for demonstration
pub fn create_sighash_message(locking_script: &ScriptBuf, value: Amount) -> Message {
    let mut engine = sha256::HashEngine::default();
    engine.input(&locking_script.to_bytes());
    engine.input(&value.to_sat().to_le_bytes());
    let digest = sha256::Hash::from_engine(engine);
    Message::from_digest(digest.to_byte_array())
}

/// Workflow
///     Alice creates $\phi(X)$ = Y, and sends to Bob
///     Alice creates output $\delta$, and sends to Bob (include description)
///     If Alice publish data Q for X, if $\phi(Q) == 0$, Bob can punish.
///     Or after a delay, Alice can refund
pub fn build_phi_eval_presigned_tx(
    pk_musig: PublicKey,
    pk_prover: PublicKey,
    y0_hash: &[u8; 32],
    csv_n: u16,
    input_txid: Txid,
    input_vout: u32,
    receiver: Address,
    network: Network,
) -> Result<(Transaction, Address)> {
    let both_failgate_script = build_failgate_script(pk_musig, y0_hash);
    let alice_refund_script = build_refund_script(pk_prover, csv_n);

    let secp = secp256k1::Secp256k1::new();

    let mut builder = TaprootBuilder::new()
        .add_leaf(0, both_failgate_script.clone())?
        .add_leaf(0, alice_refund_script.clone())?;

    let xonly_pubkey = XOnlyPublicKey::from(pk_musig);
    let spend_info = builder.finalize(&secp, xonly_pubkey).unwrap();

    let taproot_output_key = spend_info.output_key();
    let taproot_addr = Address::p2tr_tweaked(taproot_output_key, network);
    println!("🔐 Taproot script address: {}", taproot_addr);

    let amount_wo_fee = Amount::from_sat(10000);
    // construct tx
    let assert_tx = build_spending_tx(input_txid, input_vout, receiver, amount_wo_fee);

    // fill in witness

    Ok((assert_tx, taproot_addr))
}

pub fn build_input_commit_tx(
    labels: &[([u8; 32], [u8; 32])],
    pk_musig: PublicKey,
    pk_verifier: PublicKey,
    cltv_value: i64,
    input_txid: Txid,
    input_vout: u32,
    phi_address: Address,
    network: Network,
) -> Result<Transaction> {
    let commit_input_scripts = build_inputs_script(labels, pk_musig);
    let commit_timeout_script = build_commit_timeout_script(pk_verifier,  cltv_value);
    //let both_failgate_script = build_failgate_script(pk_musig, y0_hash);
    //let alice_refund_script = build_refund_script(pk_prover, csv_n);

    let secp = secp256k1::Secp256k1::new();

    let mut builder = TaprootBuilder::new()
        .add_leaf(0, commit_input_scripts.clone())?
        .add_leaf(0, commit_timeout_script.clone())?;

    let xonly_pubkey = XOnlyPublicKey::from(pk_musig);
    let spend_info = builder.finalize(&secp, xonly_pubkey).unwrap();

    let taproot_output_key = spend_info.output_key();
    let taproot_addr = Address::p2tr_tweaked(taproot_output_key, network);
    println!("🔐 Taproot script address: {}", taproot_addr);

    let amount_wo_fee = Amount::from_sat(10000);
    // construct tx
    let commit_input_tx = build_spending_tx(input_txid, input_vout, phi_address, amount_wo_fee);

    // fill in witness

    Ok(commit_input_tx)
}

fn build_spending_tx(
    input_txid: Txid,
    input_vout: u32,
    destination: Address,
    amount_wo_fee_sat: Amount,
) -> Transaction {
    let outpoint = OutPoint {
        txid: input_txid,
        vout: input_vout,
    };

    let txin = TxIn {
        previous_output: outpoint,
        script_sig: ScriptBuf::new(), // empty for P2WSH
        sequence: Sequence::from_consensus(0xffffffff),
        witness: Witness::new(), // to be filled after signing
    };

    let txout = TxOut {
        value: amount_wo_fee_sat,
        script_pubkey: destination.script_pubkey(),
    };

    Transaction {
        version: Version::TWO,
        lock_time: LockTime::ZERO,
        input: vec![txin],
        output: vec![txout],
    }
}

pub fn build_delbrag_tx(input_txid: Txid, input_vout: u32, network: Network) -> Result<Transaction> {
    let sk_signers = crate::musig2::generate_keys::<2>(); // prover, verifier
    let pk_signers: Vec<musig2::secp256k1::PublicKey> =
        sk_signers.iter().map(|key| key.1).collect::<Vec<_>>();
    let agg_ctx = musig2::KeyAggContext::new(pk_signers.clone())?;
    let pk_signer: musig2::secp256k1::PublicKey = agg_ctx.aggregated_pubkey();

    let receiver = Address::from_str("tb1qfpfy0hhzpax6xkjz9y0ns6hdj36kp04geatuw0").unwrap().require_network(network)?;

    let y0_preimage = b"some-secret-y0";
    let y0_hash = sha256::Hash::hash(y0_preimage);

    let csv_n = 1;

    let (phi_eval_tx, phi_tr_address) =  build_phi_eval_presigned_tx(
        inner_from(&pk_signer),
        inner_from(&pk_signers[0]),
        &y0_hash.to_byte_array(),
        csv_n,
        input_txid,
        input_vout,
        receiver,
        network,
    )?;

    let labels = [];
    let ctlv_value = 10;

    let input_txid = phi_eval_tx.compute_txid();
    let input_vout = 0;

    let commit_tx = build_input_commit_tx(
        &labels,
        inner_from(&pk_signer),
        inner_from(&pk_signers[0]),
        ctlv_value,
        input_txid,
        input_vout,
        phi_tr_address,
        network,
    )?;
    
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musig2::simulate_musig2;
    use crate::utils::inner_from;
    use bitcoin::script::PushBytesBuf;
    use bitcoin_script::script;
    use bitvm::{execute_script, execute_script_buf};
    use secp256k1::constants::MESSAGE_SIZE;
    use serde::Serialize;

    #[test]
    fn test_single_input_script() {
        let preimage_x0 = b"label 1";
        let expected_x0_hash = sha256::Hash::hash(preimage_x0);

        let preimage_x1 = b"label 2";
        let expected_x1_hash = sha256::Hash::hash(preimage_x1);

        for input in [preimage_x0, preimage_x1] {
            println!("input {input:?}");
            let wtns = Builder::new().push_slice(input).into_script();
            let mut script_buf = wtns.to_bytes();

            let script = build_input_script(
                &expected_x0_hash.to_byte_array(),
                &expected_x1_hash.to_byte_array(),
            );
            let locking_script = combine_scripts(&[script, script! {OP_TRUE}.compile()]);
            script_buf.extend(locking_script.to_bytes());
            let res = execute_script_buf(ScriptBuf::from_bytes(script_buf));
            println!("error: {:?}", res.error);
            println!("final_stack: {:?}", res.final_stack);
            println!("last_opcode: {:?}", res.last_opcode);
            assert!(res.success);
        }
    }

    #[test]
    fn test_commit_inputs_script_builds() {
        let sk_signers = crate::musig2::generate_keys::<2>(); // prover, verifier
        let pk_signers: Vec<musig2::secp256k1::PublicKey> =
            sk_signers.iter().map(|key| key.1).collect::<Vec<_>>();
        let agg_ctx = musig2::KeyAggContext::new(pk_signers).unwrap();
        let pk_signer: musig2::secp256k1::PublicKey = agg_ctx.aggregated_pubkey();

        // Sample H(Xi₀), H(Xi₁) hashes
        let labels = vec![
            (
                sha256::Hash::hash(b"zero").to_byte_array(),
                sha256::Hash::hash(b"one").to_byte_array(),
            ),
            (
                sha256::Hash::hash(b"two").to_byte_array(),
                sha256::Hash::hash(b"three").to_byte_array(),
            ),
        ];

        let commit_inputs =
            build_inputs_script(&labels, inner_from(pk_signer));
        let locking_script = combine_scripts(&[commit_inputs, script! {OP_TRUE}.compile()]);

        // Simulate a message to multi-sig
        let msg = Message::from_digest_slice(&[0xab; MESSAGE_SIZE]).unwrap();
        let sig_der = simulate_musig2(&sk_signers, &msg).unwrap();
        let mut sig_der = sig_der.serialize().to_vec();
        sig_der.push(0x01); // SIGHASH_ALL
        let sig_push = PushBytesBuf::try_from(sig_der).expect("Invalid signature for push");

        let wtns = Builder::new()
            .push_slice(&sig_push)
            .push_slice(b"three") // or two
            .push_slice(b"one") // or zero
            .into_script();

        let mut wtns = wtns.to_bytes();
        wtns.extend(locking_script.to_bytes());

        let exec_script = ScriptBuf::from_bytes(wtns);
        let res = execute_script_buf(exec_script);

        println!("error: {:?}", res.error);
        println!("final_stack: {:?}", res.final_stack);
        println!("last_opcode: {:?}", res.last_opcode);
        assert!(res.success);
    }

    #[test]
    fn test_build_failgate_script() {
        let sk_signers = crate::musig2::generate_keys::<2>(); // prover, verifier
        let pk_signers: Vec<musig2::secp256k1::PublicKey> =
            sk_signers.iter().map(|key| key.1).collect::<Vec<_>>();
        let agg_ctx = musig2::KeyAggContext::new(pk_signers).unwrap();
        let pk_signer: musig2::secp256k1::PublicKey = agg_ctx.aggregated_pubkey();

        // Simulate Y0 preimage and hash
        let y0_preimage = b"some-secret-y0";
        let y0_hash = sha256::Hash::hash(y0_preimage);

        let output_script = build_failgate_script(inner_from(pk_signer), &y0_hash.to_byte_array());

        let locking_script = combine_scripts(&[output_script, script! {OP_TRUE}.compile()]);

        let msg = Message::from_digest_slice(&[0xab; MESSAGE_SIZE]).unwrap();
        let sig_der = simulate_musig2(&sk_signers, &msg).unwrap();
        let mut sig_der = sig_der.serialize().to_vec();
        sig_der.push(0x01); // SIGHASH_ALL
        let sig_push = PushBytesBuf::try_from(sig_der).expect("Invalid signature for push");

        let wtns = Builder::new().push_slice(&sig_push).push_slice(y0_preimage).into_script();

        let mut wtns = wtns.to_bytes();
        wtns.extend(locking_script.to_bytes());

        let exec_script = ScriptBuf::from_bytes(wtns);
        let exec_info = execute_script_buf(exec_script);

        println!("error {:?}", exec_info.error);
        println!("stack {:?}", exec_info.final_stack);
        assert!(exec_info.success);
    }

}
