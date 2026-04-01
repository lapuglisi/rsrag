use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CollectionExistsRequest, Query, QueryPointsBuilder, ScoredPoint};

pub const QDRANT_QUERY_DEFAULT_THRESHOLD: f32 = 0.7;

#[derive(Clone)]
pub struct QdrantQuery {
  pub qdrant: Option<QdrantEngine>,
  pub collection: Option<String>,
  pub embeddings: Option<Vec<f32>>,
  pub threshold: f32,
}

impl Default for QdrantQuery {
  fn default() -> Self {
    Self {
      qdrant: None,
      collection: None,
      embeddings: None,
      threshold: QDRANT_QUERY_DEFAULT_THRESHOLD,
    }
  }
}

impl QdrantQuery {
  pub fn new(engine: &QdrantEngine) -> Self {
    Self::default().with_qdrant(engine)
  }

  fn with_qdrant(mut self, q: &QdrantEngine) -> Self {
    self.qdrant = Some(q.to_owned());
    self
  }

  pub fn with_collection(mut self, c: String) -> Self {
    self.collection = Some(c);
    self
  }

  pub fn with_embeddings(mut self, e: Vec<f32>) -> Self {
    self.embeddings = Some(e);
    self
  }

  pub fn with_threshold(mut self, t: f32) -> Self {
    self.threshold = t;
    self
  }

  pub async fn run(
    self,
  ) -> Result<Vec<ScoredPoint>, Box<dyn std::error::Error + Send + Sync + 'static>> {
    let qdrant = self.qdrant.ok_or_else(|| "qdrant engine not initialized")?;
    let client = qdrant
      .client
      .ok_or_else(|| "qdrant client is not initialized")?;
    let collection = self.collection.ok_or_else(|| "no collection defined")?;
    let embeddings = self.embeddings.ok_or_else(|| "no embeddings provided")?;

    if !client
      .collection_exists(CollectionExistsRequest {
        collection_name: String::from(&collection),
      })
      .await
      .unwrap_or(false)
    {
      log::error!("qdrant: collection {} does not exist!", collection);
      return Err(format!("qdrant: collection {} does not exist", collection))?;
    }

    let mut qp = QueryPointsBuilder::new(collection)
      .query(Query::new_nearest(embeddings))
      .with_payload(true)
      .score_threshold(self.threshold);

    if qdrant.limit > 0 {
      log::info!("qdrant search: using limit {}", qdrant.limit);
      qp = qp.limit(qdrant.limit);
    }

    let res = match client.query(qp).await {
      Ok(v) => {
        log::info!("got qdrant result: {:?}", v);
        v
      }
      Err(e) => {
        log::error!("qdrant: error {}", e);
        return Err(e)?;
      }
    };

    if res.result.len() > 0 {
      Ok(res.result)
    } else {
      Err("query yielded no points")?
    }
  }
}

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
