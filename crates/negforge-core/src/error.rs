use thiserror::Error;

#[derive(Debug, Error)]
pub enum NegForgeError {
    #[error("self-consistent loop did not converge after {iterations} iterations (residual {residual:.3e}, tolerance {tolerance:.3e})")]
    NotConverged {
        iterations: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type Result<T> = std::result::Result<T, NegForgeError>;
