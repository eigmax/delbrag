/// This is special for conversion of SecretKey and PublicKey between ::musig2::secp256k and ::secp256k1
use serde::{Serialize, de::DeserializeOwned};
pub fn inner_from<F: Serialize, T: DeserializeOwned>(from: F) -> T {
    let value = serde_json::to_value(&from).unwrap();
    serde_json::from_value(value).unwrap()
}
