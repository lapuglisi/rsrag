use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::time::Duration;

//
// Llama completion
//
#[derive(Serialize, Debug)]
struct LlamaCompletionRequest {
  model: String,
  messages: Vec<LlamaCompletionMessages>,
  temperature: f32,
  n_predict: i32,
  top_k: i32,
  top_p: f32,
  stream: bool,
}

impl Default for LlamaCompletionRequest {
  fn default() -> Self {
    Self {
      model: String::new(),
      messages: Vec::new(),
      temperature: 0.8,
      n_predict: -1,
      top_k: 40,
      top_p: 0.9,
      stream: true,
    }
  }
}

impl LlamaCompletionRequest {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn with_model(mut self, m: &str) -> Self {
    self.model = m.into();
    self
  }

  pub fn messages(self) -> Vec<LlamaCompletionMessages> {
    self.messages
  }

  pub fn with_temperature(mut self, t: f32) -> Self {
    self.temperature = t;
    self
  }

  pub fn with_n_predict(mut self, p: i32) -> Self {
    self.n_predict = p;
    self
  }

  pub fn with_top_k(mut self, k: i32) -> Self {
    self.top_k = k;
    self
  }
}

#[derive(Serialize, Debug)]
struct LlamaCompletionMessages {
  role: String,
  content: String,
}

impl Default for LlamaCompletionMessages {
  fn default() -> Self {
    Self {
      role: "user".to_string(),
      content: String::new(),
    }
  }
}

//
//
//

#[derive(Serialize, Deserialize)]
struct LlamaEmbedRequest {
  input: String,
}

//
// Llama Embedding Response
//
#[derive(Serialize, Debug)]
pub struct EmbedResponse {
  pub model: String,
  pub embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct LlamaEmbedResponseData {
  index: u32,
  object: String,
  embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct LlamaEmbedResponseUsage {
  prompt_tokens: u32,
  total_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct LlamaEmbedResponse {
  model: String,
  object: String,
  usage: LlamaEmbedResponseUsage,
  data: Vec<LlamaEmbedResponseData>,
}
////////////////////////////

#[derive(Clone, Default)]
pub struct LlamaEngine {
  embed_server: String,
  llama_server: String,
  rerank_server: String,
  chat_model: String,
  embed_model: String,
  rerank_model: String,
}

impl<'l> LlamaEngine {
  pub fn new() -> Self {
    LlamaEngine::default()
  }

  pub fn with_llama_server(mut self, s: &str) -> Self {
    self.llama_server = s.into();
    self
  }

  pub fn with_embed_server(mut self, s: &str) -> Self {
    self.embed_server = s.into();
    self
  }

  pub fn with_rerank_server(mut self, s: &str) -> Self {
    self.rerank_server = s.into();
    self
  }

  pub fn with_chat_model(mut self, s: &str) -> Self {
    self.chat_model = s.into();
    self
  }

  pub fn with_embed_model(mut self, s: &str) -> Self {
    self.embed_model = s.into();
    self
  }

  pub fn with_rerank_model(mut self, s: &str) -> Self {
    self.rerank_model = s.into();
    self
  }

  pub fn print(self) {
    println!("--- LlamaEngine ---");
    println!("embed_server ..... {}", self.embed_server);
    println!("llama_server ..... {}", self.llama_server);
    println!("rerank_server .... {}", self.rerank_server);
    println!("chat_model ....... {}", self.chat_model);
    println!("embed_model ...... {}", self.embed_model);
    println!("rerank_model ..... {}", self.rerank_model);
  }

  // Endpoints implementation
  pub async fn get_embeddings(
    self,
    input: &str,
  ) -> Result<EmbedResponse, Box<dyn std::error::Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);
    let er = LlamaEmbedRequest {
      input: input.into(),
    };

    let json = serde_json::to_string(&er)?;

    println!("sending request to {}", url);
    println!("json is {}", json);

    let client = Client::new()
      .post(url)
      .header("Content-Type", "application/json")
      .body(json)
      .timeout(Duration::from_secs(5))
      .send()
      .await?;

    let body = client.text().await?;
    let erp: LlamaEmbedResponse = serde_json::from_str(&body)?;

    let embed = EmbedResponse {
      model: erp.model,
      embedding: erp.data[0].embedding.clone(),
    };

    println!("returning response: {:?}", embed);

    Ok(embed)
  }

  pub async fn get_completion() -> String {
    String::from("Busco completion com rag")
  }
}
