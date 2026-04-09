use qdrant_client::Qdrant;
use qdrant_client::qdrant::{
  CollectionExistsRequest, Fusion, PrefetchQuery, PrefetchQueryBuilder, Query, QueryPointsBuilder,
  ScoredPoint,
};

pub const QDRANT_QUERY_DEFAULT_THRESHOLD: f32 = 0.9;
pub const QDRANT_QUERY_DEFAULT_LIMIT: u64 = 5;

#[derive(Clone)]
pub struct QdrantQuery {
  pub qdrant: Option<QdrantEngine>,
  pub collection: Option<String>,
  pub threshold: f32,
  pub prefetchs: Option<Vec<PrefetchQuery>>,
  pub query: Query,
  pub default_vector: Option<String>,
  pub limit: u64,
  pub cancel: bool,
}

impl Default for QdrantQuery {
  fn default() -> Self {
    Self {
      qdrant: None,
      collection: None,
      threshold: QDRANT_QUERY_DEFAULT_THRESHOLD,
      prefetchs: None,
      query: Query::default(),
      default_vector: None,
      limit: QDRANT_QUERY_DEFAULT_LIMIT,
      cancel: false,
    }
  }
}

#[allow(dead_code)]
impl QdrantQuery {
  pub fn new(engine: &QdrantEngine) -> Self {
    Self::default()
      .with_qdrant(engine)
      .with_collection(&engine.collection)
  }

  fn with_qdrant(mut self, q: &QdrantEngine) -> Self {
    self.qdrant = Some(q.to_owned());
    self
  }

  pub fn with_collection(mut self, c: &str) -> Self {
    self.collection = Some(String::from(c));
    self
  }

  pub fn add_prefetch(mut self, using: &str, q: Vec<f32>) -> Self {
    let mut pfs = self.prefetchs.unwrap_or(Vec::new());

    pfs.push(
      PrefetchQueryBuilder::default()
        .using(using)
        .query(q)
        .build(),
    );

    self.prefetchs = Some(pfs);

    self
  }

  pub fn add_sparse_prefetch(mut self, using: &str, text: &str, model: Option<&str>) -> Self {
    let mut pfs = self.prefetchs.unwrap_or(Vec::new());
    let doc = qdrant_client::qdrant::Document::new(text, model.unwrap_or("qdrant/bm25"));

    pfs.push(
      PrefetchQueryBuilder::default()
        .using(using)
        .query(Query::new_nearest(doc))
        .build(),
    );

    self.prefetchs = Some(pfs);

    self
  }

  pub fn query(mut self, using: &str, e: Vec<f32>) -> Self {
    self.default_vector = Some(String::from(using));
    self.query = Query::new_nearest(e);
    self
  }

  pub fn with_fusion_dbsf(mut self) -> Self {
    self.default_vector = None;
    self.query = Query::new_fusion(Fusion::Dbsf);
    self
  }

  pub fn with_fusion_rrf(mut self) -> Self {
    self.default_vector = None;
    self.query = Query::new_fusion(Fusion::Rrf);
    self
  }

  pub fn with_threshold(mut self, t: f32) -> Self {
    self.threshold = t;
    self
  }

  pub fn with_limit(mut self, l: u64) -> Self {
    self.limit = l;
    self
  }

  pub fn cancel(mut self, b: bool) -> Self {
    self.cancel = b;
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

    if self.cancel {
      return Err("query is cancelled.")?;
    }

    let mut qp = QueryPointsBuilder::new(collection)
      .query(self.query)
      .with_payload(true)
      .timeout(300)
      .score_threshold(self.threshold);

    if self.limit > 0 {
      log::info!("qdrant search: using limit {}", self.limit);
      qp = qp.limit(self.limit);
    }

    if let Some(v) = self.default_vector {
      qp = qp.using(v);
    }

    if let Some(prefetchs) = self.prefetchs {
      for p in prefetchs {
        qp = qp.add_prefetch(p);
      }
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

#[allow(dead_code)]
#[derive(Clone)]
pub struct QdrantEngine {
  pub limit: u64,
  pub client: Option<Qdrant>,
  pub collection: String,
}

#[allow(dead_code)]
impl QdrantEngine {
  pub fn new(url: String) -> Self {
    Self {
      client: match Qdrant::from_url(&url).build() {
        Ok(v) => Some(v),
        Err(_) => None,
      },
      limit: 0,
      collection: String::new(),
    }
  }

  pub fn with_collection(mut self, collection: &str) -> Self {
    self.collection = String::from(collection);
    self
  }

  pub fn limit(mut self, l: u64) -> Self {
    self.limit = l;
    self
  }
}

impl Default for QdrantEngine {
  fn default() -> Self {
    Self {
      client: None,
      limit: 0,
      collection: String::new(),
    }
  }
}
