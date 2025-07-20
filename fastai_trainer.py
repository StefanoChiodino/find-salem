#!/usr/bin/env python3
"""
FastAI Salem Classifier - Based on fastbook Chapter 1 approach
Uses ImageDataLoaders, pre-trained CNN models, and transfer learning.
"""

from fastai.vision.all import *
from pathlib import Path
import pandas as pd

def setup_data():
    """Setup data loaders using FastAI's ImageDataLoaders (fastbook Chapter 1 approach)"""
    print("🐱 Setting up FastAI Salem Classifier")
    print("=" * 50)
    
    # Define data path
    data_path = Path("data")
    
    if not data_path.exists():
        print("❌ Data directory not found!")
        return None
    
    # Create ImageDataLoaders from folder structure (fastbook approach)
    # This automatically handles train/validation split, data augmentation, etc.
    dls = ImageDataLoaders.from_folder(
        data_path,
        train='train',      # Training folder
        valid='test',       # Use test folder as validation (fastbook pattern)
        item_tfms=Resize(224),  # Resize to 224x224 (standard for pretrained models)
        batch_tfms=aug_transforms(size=224, min_scale=0.75),  # Data augmentation
        bs=32  # Batch size
    )
    
    print(f"✅ Data loaded successfully!")
    print(f"📊 Classes: {dls.vocab}")
    print(f"📸 Training batches: {len(dls.train)}")
    print(f"📸 Validation batches: {len(dls.valid)}")
    
    # Show sample batch (fastbook visualization)
    print("\n📋 Sample batch:")
    dls.show_batch(max_n=8, figsize=(10, 8))
    
    return dls

def create_learner(dls):
    """Create learner with pre-trained model (fastbook Chapter 1 approach)"""
    print("\n🚀 Creating FastAI Learner...")
    
    # Create CNN learner with pre-trained ResNet34 (fastbook standard)
    learn = cnn_learner(
        dls, 
        resnet34,           # Pre-trained architecture (fastbook default)
        metrics=[accuracy, error_rate]  # Track accuracy and error rate
    )
    
    print(f"✅ Created learner with {resnet34.__name__}")
    print(f"📋 Architecture: Pre-trained ResNet34")
    print(f"📊 Metrics: Accuracy, Error Rate")
    
    return learn

def find_learning_rate(learn):
    """Find optimal learning rate (fastbook Chapter 1 technique)"""
    print("\n🔍 Finding optimal learning rate...")
    
    # Learning rate finder (fastbook technique)
    lr_find = learn.lr_find(suggest_funcs=(minimum, steep, valley, slide))
    
    print(f"📈 Suggested learning rates:")
    print(f"   Minimum: {lr_find.minimum:.2e}")
    print(f"   Steep: {lr_find.steep:.2e}")
    print(f"   Valley: {lr_find.valley:.2e}")
    print(f"   Slide: {lr_find.slide:.2e}")
    
    # Use the minimum suggestion (fastbook recommendation)
    suggested_lr = lr_find.valley
    print(f"🎯 Using learning rate: {suggested_lr:.2e}")
    
    return suggested_lr

def train_model(learn, lr):
    """Train the model using FastAI fine-tuning approach (fastbook Chapter 1)"""
    print("\n🏋️ Training Salem classifier...")
    
    # Fine-tune pre-trained model (fastbook 2-step approach)
    print("📈 Step 1: Training head only (frozen backbone)...")
    learn.fine_tune(
        epochs=3,           # Train head for 3 epochs (fastbook default)
        base_lr=lr,         # Use suggested learning rate
        freeze_epochs=1     # Keep backbone frozen for 1 epoch
    )
    
    print("✅ Training completed!")
    
    # Show results
    learn.show_results(max_n=6, figsize=(12, 8))
    
    return learn

def evaluate_model(learn):
    """Evaluate model performance (fastbook Chapter 1 approach)"""
    print("\n📊 Model Evaluation:")
    print("=" * 30)
    
    # Get predictions on validation set
    preds, targets = learn.get_preds()
    
    # Calculate accuracy
    accuracy = (preds.argmax(dim=1) == targets).float().mean()
    print(f"🎯 Validation Accuracy: {accuracy:.1%}")
    
    # Show confusion matrix (fastbook visualization)
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(8, 8))
    
    # Show top losses (fastbook debugging technique)
    print("\n🔍 Top losses (most confused predictions):")
    interp.plot_top_losses(6, nrows=2, figsize=(12, 8))
    
    return accuracy

def save_model(learn, model_name="salem_fastai_model"):
    """Save the trained model (fastbook approach)"""
    print(f"\n💾 Saving model as '{model_name}'...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Export the model (fastbook production approach)
    learn.export(f'models/{model_name}.pkl')
    
    print(f"✅ Model saved to models/{model_name}.pkl")
    print(f"💡 Use `load_learner()` to load the model for inference")

def test_inference(model_path="models/salem_fastai_model.pkl"):
    """Test model inference (fastbook Chapter 1 approach)"""
    print("\n🧪 Testing model inference...")
    
    # Load the exported model
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    learn_inf = load_learner(model_path)
    
    # Test on demo samples
    demo_salem_dir = Path("demo_samples/salem")
    demo_other_dir = Path("demo_samples/other_cats")
    
    test_images = []
    if demo_salem_dir.exists():
        test_images.extend([(f, "Salem") for f in demo_salem_dir.glob("*.jpg")[:2]])
    if demo_other_dir.exists():
        test_images.extend([(f, "Other Cat") for f in demo_other_dir.glob("*.jpg")[:2]])
    
    if not test_images:
        print("❌ No demo images found for testing")
        return
    
    print(f"📸 Testing on {len(test_images)} sample images:")
    
    for img_path, true_label in test_images:
        try:
            # Make prediction (fastbook inference)
            pred_class, pred_idx, probs = learn_inf.predict(img_path)
            confidence = probs.max().item()
            
            print(f"\n🖼️ {img_path.name}")
            print(f"   True: {true_label}")
            print(f"   Predicted: {pred_class} ({confidence:.1%} confidence)")
            
            if str(pred_class) == true_label:
                print("   ✅ Correct")
            else:
                print("   ❌ Incorrect")
                
        except Exception as e:
            print(f"❌ Error testing {img_path.name}: {e}")

def main():
    """Main training function following fastbook Chapter 1 approach"""
    try:
        # Step 1: Setup data loaders
        dls = setup_data()
        if dls is None:
            return
        
        # Step 2: Create learner with pre-trained model
        learn = create_learner(dls)
        
        # Step 3: Find optimal learning rate
        lr = find_learning_rate(learn)
        
        # Step 4: Train the model
        learn = train_model(learn, lr)
        
        # Step 5: Evaluate performance
        accuracy = evaluate_model(learn)
        
        # Step 6: Save the model
        save_model(learn)
        
        # Step 7: Test inference
        test_inference()
        
        print(f"\n🎉 FastAI Salem classifier training complete!")
        print(f"🎯 Final validation accuracy: {accuracy:.1%}")
        print(f"💡 Model saved and ready for use in web interface")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
