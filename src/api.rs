use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use serde::{Deserialize, Serialize};

use crate::engine::{
  llama::{
    LLAMA_DEFAULT_NPREDICT, LLAMA_DEFAULT_STREAM, LLAMA_DEFAULT_TEMPERATURE, LLAMA_DEFAULT_TOP_K,
    LLAMA_DEFAULT_TOP_P, LlamaCompletionMessage, LlamaCompletionRequest, LlamaEngine,
  },
  qdrant::{QDRANT_QUERY_DEFAULT_THRESHOLD, QdrantEngine, QdrantQuery},
};

const RAGAPI_DEFAULT_HOST: &str = "127.0.0.1";
const RAGAPI_DEFAULT_PORT: u16 = 9091;
const RAGAPI_LLM_SYSTEM_PROMPT: &str = "You are a helpful assistant expert in answering prompts in a RAG environment.\nAnswer the user prompt using only the provided context. If you do not know the answer, simply state that you don't know.";

macro_rules! RAGAPI_LLM_USER_PROMPT {
  () => {
    "Use the provided context to answer the question.\n\nContext: {}\n\nQuestion: {}"
  };
}

// Structs for json requests
#[derive(Deserialize, Debug)]
struct EmbedRequest {
  model: String,
  input: String,
}

#[derive(Serialize, Debug)]
pub struct EmbedResponse {
  pub model: String,
  pub embeddings: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(default)]
pub struct CompletionRequest {
  pub model: String,
  pub prompt: String,
  pub temperature: Option<f32>,
  pub stream: Option<bool>,
  pub n_predict: Option<i32>,
  pub top_k: Option<u32>,
  pub top_p: Option<f32>,
  pub threshold: Option<f32>,
}

impl Default for CompletionRequest {
  fn default() -> Self {
    Self {
      model: String::new(),
      prompt: String::new(),
      temperature: Some(LLAMA_DEFAULT_TEMPERATURE),
      stream: Some(LLAMA_DEFAULT_STREAM),
      n_predict: Some(LLAMA_DEFAULT_NPREDICT),
      top_k: Some(LLAMA_DEFAULT_TOP_K),
      top_p: Some(LLAMA_DEFAULT_TOP_P),
      threshold: Some(QDRANT_QUERY_DEFAULT_THRESHOLD),
    }
  }
}

#[derive(Clone)]
pub struct RagApiState {
  llama: LlamaEngine,
  qdrant: QdrantEngine,
}

pub struct RagApi<'r> {
  host: &'r str,
  port: u16,
  llama: Option<LlamaEngine>,
  qdrant: Option<QdrantEngine>,
}

impl<'r> RagApi<'r> {
  pub fn new() -> Self {
    Self {
      host: RAGAPI_DEFAULT_HOST,
      port: RAGAPI_DEFAULT_PORT,
      llama: None,
      qdrant: None,
    }
  }

  pub fn with_listen_host(mut self, address: &'r str) -> Self {
    self.host = address;
    self
  }

  pub fn with_listen_port(mut self, port: u16) -> Self {
    self.port = port;
    self
  }

  pub fn with_llama_engine(mut self, llama: LlamaEngine) -> Self {
    self.llama = Some(llama);
    self
  }

  pub fn with_qdrant_engine(mut self, q: QdrantEngine) -> Self {
    self.qdrant = Some(q);
    self
  }

  pub async fn listen(self) -> Result<(), Box<dyn std::error::Error>> {
    if self.llama.is_none() {
      Err("[rsrag::api::listen] no llama engine configured")?
    }

    if self.qdrant.is_none() {
      Err("[rsrag::api::listen] no qdrant engine configured.")?
    }

    let st = RagApiState {
      llama: self.llama.unwrap(),
      qdrant: self.qdrant.unwrap(),
    };

    let app: Router = Router::new()
      .route("/api/embeddings", post(api_embeddings))
      .route("/api/completion", post(api_completion))
      .with_state(st);

    let address = format!("{}:{}", self.host, self.port);

    let listener = tokio::net::TcpListener::bind(&address)
      .await
      .map_err(|err| {
        format!(
          "[rsrag::api]: Could not bind to address {}: {}",
          address,
          err.to_string()
        )
      })?;

    println!("[rsrag::api] listening on '{}'...", address);
    log::info!("[rsrag::api] listening on '{}'...", address);

    let _ = axum::serve(listener, app).await;

    Ok(())
  }
}

fn get_qdrant_colletion(model: &str) -> String {
  model
    .chars()
    .map(|c| if c.is_ascii_punctuation() { '-' } else { c })
    .collect()
}

async fn api_embeddings(
  State(st): State<RagApiState>,
  Json(payload): Json<EmbedRequest>,
) -> impl IntoResponse {
  log::debug!("[api_embeddings] payload is {:?}", payload);

  let resp = match st.llama.get_embeddings(&payload.input).await {
    Ok(v) => (
      StatusCode::OK,
      serde_json::to_string(&v).unwrap_or("{}".to_string()),
    ),
    Err(e) => (
      StatusCode::INTERNAL_SERVER_ERROR,
      format!("embeddings error: {}", e),
    ),
  };

  resp.into_response()
}

async fn api_completion(
  State(st): State<RagApiState>,
  Json(payload): Json<CompletionRequest>,
) -> impl IntoResponse {
  log::debug!("[api_completion] payload is {:?}", payload);

  get_rag_completion(st, payload).await
}

async fn get_rag_completion(state: RagApiState, req: CompletionRequest) -> impl IntoResponse {
  // get embeddings for prompt
  let llama = state.llama.clone();
  let mut messages: Vec<LlamaCompletionMessage> = Vec::new();

  if state.qdrant.client.is_none() {
    return (
      StatusCode::INTERNAL_SERVER_ERROR,
      String::from("no qdrant client confifured"),
    )
      .into_response();
  }

  let er: EmbedResponse = match llama.get_embeddings(&req.prompt).await {
    Ok(v) => v,
    Err(e) => {
      return (StatusCode::INTERNAL_SERVER_ERROR, format!("{}", e)).into_response();
    }
  };

  // Do qdrant similarity search
  let collection: String = get_qdrant_colletion(&er.model);
  log::info!("using qdrant collection {}", &collection);

  match QdrantQuery::new(state.qdrant)
    .with_collection(collection)
    .with_embeddings(er.embeddings)
    .with_threshold(req.threshold.unwrap())
    .run()
    .await
  {
    Ok(res) => {
      log::info!("qdrant::query_points yielded {:?}", res);

      messages.push(
        LlamaCompletionMessage::new()
          .with_role("system")
          .with_content(RAGAPI_LLM_SYSTEM_PROMPT),
      );

      let mut context: String = String::new();
      res.iter().for_each(|item| {
        if let Some(v) = item.payload["source"].as_str() {
          context.push_str(v);
          context.push_str("\n");
        }
        log::info!("inside foreach");
      });

      let user_msg = format!(RAGAPI_LLM_USER_PROMPT!(), context, req.prompt);
      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(&user_msg),
      );
    }
    Err(e) => {
      log::info!("qdrant yielded no results {}.", e);
      messages.push(
        LlamaCompletionMessage::new()
          .with_role("user")
          .with_content(&req.prompt),
      );
    }
  }

  // Prepare a completion request
  let lcr = LlamaCompletionRequest::new()
    .with_model(&req.model)
    .with_temperature(req.temperature)
    .with_top_k(req.top_k)
    .with_top_p(req.top_p)
    .with_n_predict(req.n_predict)
    .stream(req.stream)
    .append_messages(messages);

  log::debug!("request llama with {:?}", lcr);

  state.llama.get_completion(lcr).await.into_response()
}
