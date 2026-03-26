use qdrant_client::Qdrant;
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::{Query, QueryPointsBuilder};

#[derive(Clone)]
pub struct QdrantEngine {
  pub limit: u64,
  pub client: Option<Qdrant>,
}

impl QdrantEngine {
  pub fn new(url: String) -> Self {
    Self {
      client: match Qdrant::from_url(&url).build() {
        Ok(v) => Some(v),
        Err(_) => None,
      },
      limit: 0,
    }
  }
}

impl Default for QdrantEngine {
  fn default() -> Self {
    Self {
      client: None,
      limit: 0,
    }
  }
}
