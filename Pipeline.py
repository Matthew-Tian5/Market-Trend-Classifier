from ingestion import run_ingestion
from features  import run_feature_engineering
from train     import train_model
from evaluate  import evaluate
 
 
if __name__ == "__main__":
    print("=" * 50)
    print("Stage 1 — Ingestion")
    print("=" * 50)
    run_ingestion()
 
    print("\n" + "=" * 50)
    print("Stage 2 — Feature Engineering")
    print("=" * 50)
    run_feature_engineering()
 
    print("\n" + "=" * 50)
    print("Stage 3 — Training")
    print("=" * 50)
    train_model()
 
    print("\n" + "=" * 50)
    print("Stage 4 — Evaluation")
    print("=" * 50)
    evaluate()