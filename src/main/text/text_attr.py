import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

from captum.attr import IntegratedGradients, GradientShap, DeepLift, Occlusion, FeatureAblation, Saliency, Lime, KernelShap, visualization
from ryy.code.latec.src.nlp_model_emotion.lstm_main import vocab_path
from train_text_model import preprocess_data_agnews,preprocess_data_imdb, create_data_loader, TransformerClassifier,LSTMClassifier
from captum.attr import TokenReferenceBase



class ModelWrapper(nn.Module):
    """
    Model wrapper for Captum explanation methods
    This wrapper ensures input is word IDs, output is logits, and supports embedding layer hooks
    """
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        # Register forward hook for embedding layer to let Captum track gradients
        self._register_embedding_hook()
    
    def _register_embedding_hook(self):
        """Register forward hook for embedding layer"""
        def embedding_forward_hook(module, input, output):
            # Save embedding output for Captum use
            self.embedding_output = output
        
        # Register hook
        self.model.embedding.register_forward_hook(embedding_forward_hook)
    
    def forward(self, input_ids):
        # Ensure input is long integer (word IDs)
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        
        # Directly call original model
        return self.model(input_ids)

class EmbeddingWrapper_Transformer(nn.Module):
    """
    Word embedding wrapper for gradient methods
    Convert word IDs to word embeddings, then compute gradients
    """
    def __init__(self, model):
        super(EmbeddingWrapper_Transformer, self).__init__()
        self.model = model
        # Get word embedding layer
        self.embedding = model.embedding
    
    def forward(self, embeddings):
        # Input is word embeddings, need to find corresponding word IDs
        # Here we directly use word embeddings, bypassing embedding layer
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Create positional encoding
        positions = torch.arange(0, seq_len, device=embeddings.device).unsqueeze(0)
        pos_embedding = self.model.pos_embedding(positions)
        
        # Add positional encoding
        x = embeddings + pos_embedding
        
        # Pass through transformer encoder
        out = self.model.transformer_encoder(x)
        out = out.mean(dim=1)  # Average pooling
        out = self.model.dropout(out)
        out = self.model.fc(out)
        return out

class EmbeddingWrapper_LSTM(nn.Module):
    """
    Word embedding wrapper for gradient methods
    Convert word IDs to word embeddings, then compute gradients
    """
    def __init__(self, model):
        super(EmbeddingWrapper_LSTM, self).__init__()
        self.model = model
        # Get word embedding layer
        self.embedding = model.embedding
    
    def forward(self, embeddings):
        # Input is word embeddings, directly process through LSTM
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Temporarily set to training mode to support gradient computation
        was_training = self.model.training
        self.model.train()
        
        try:
            # Pass through LSTM layer
            out, _ = self.model.lstm(embeddings)
            out = out.mean(dim=1)  # Average pooling
            out = self.model.fc1(out)
            out = self.model.relu(out)
            out = self.model.fc2(out)
            return out
        finally:
            # Restore original mode
            if not was_training:
                self.model.eval()


def load_model(model_path, vocab_size, embed_size, num_heads, hidden_dim, num_layers, output_size, max_len, device):
    """Load and wrap model"""
    # Create model (using transformer_main.py structure)
    model = TransformerClassifier(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_size=output_size,
        max_len=max_len
    )
    # model = LSTMClassifier(
    #     vocab_size=vocab_size,
    #     embed_size=embed_size,
    #     hidden_size=hidden_dim,
    #     output_size=output_size,
    #     num_layers=num_layers
    # )
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    # Wrap model
    wrapped_model = ModelWrapper(model)
    wrapped_model.to(device)
    wrapped_model.eval()
    
    # Create embedding wrapper (for gradient methods)
    embedding_wrapper = EmbeddingWrapper_Transformer(model)
    # embedding_wrapper_lstm = EmbeddingWrapper_LSTM(model)
    embedding_wrapper.to(device)
    embedding_wrapper.eval()
    
    return wrapped_model, embedding_wrapper, model

def get_explainer(method, model, embedding_model=None):
    """Get explainer"""
    if method == 'IntegratedGradients':
        return IntegratedGradients(model)
    elif method == 'GradientShap':
        return GradientShap(model)
    elif method == 'DeepLift':
        return DeepLift(model)
    elif method == 'Occlusion':
        return Occlusion(model)
    elif method == 'FeatureAblation':
        return FeatureAblation(model)
    elif method == 'Saliency':
        return Saliency(model)
    elif method == 'Lime':
        return Lime(model)
    elif method == 'KernelShap':
        return KernelShap(model)
    else:
        raise ValueError(f"Unknown explanation method: {method}")

def explain_batch(
    data_loader,
    model,
    embedding_model=None,
    PAD_IND=0,  
    method='Saliency',
    device='cpu',
    max_batches=None,
    word2idx=None,
    batch_size=8  
):
    """
    Batch explanation function - supports gradient methods
    
    Args:
        data_loader: Data loader
        model: Wrapped model (for non-gradient methods)
        embedding_model: Embedding wrapper (for gradient methods)
        method: Explanation method
        device: Device
        max_batches: Maximum number of batches
        word2idx: Word to index mapping (for visualization)
        batch_size: Batch size (reduce memory usage)
    """
    model.eval()
    if embedding_model:
        embedding_model.eval()
    
    # For gradient methods, use EmbeddingWrapper (handle embedding input)
    # For non-gradient methods, use original model
    if method in ['IntegratedGradients', 'GradientShap', 'DeepLift', 'Saliency']:
        explainer = get_explainer(method, embedding_model)
    else:
        explainer = get_explainer(method, model)
    
    
    token_reference = TokenReferenceBase(reference_token_idx=PAD_IND)
    all_attributions = []
    all_labels = []
    all_texts = []

    total_batches = len(data_loader) if hasattr(data_loader, '__len__') else None
    for batch_idx, (input_ids, labels) in enumerate(data_loader):
        if batch_idx % 20 == 0:  # Print progress every 20 batches
            print(f"Processing batch {batch_idx+1}/{total_batches}")
        
        # Limit batch size to reduce memory usage
        if input_ids.size(0) > batch_size:
            # Only process first batch_size samples
            input_ids = input_ids[:batch_size].to(device)
            labels = labels[:batch_size].to(device)
        else:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
        
        # Ensure input is long integer
        if input_ids.dtype != torch.long:
            input_ids = input_ids.long()
        

        sequence_length = input_ids.shape[1]
        reference_indices = token_reference.generate_reference(sequence_length, device=device).unsqueeze(0)

        try:
            if method in ['IntegratedGradients', 'GradientShap', 'DeepLift', 'Saliency']:
                # Gradient methods: use embedding as input
                # Get embedding
                embeddings = model.model.embedding(input_ids)
                embeddings.requires_grad_()  # Ensure differentiable
                embedding_model.train()

                # Broadcast reference_indices to same dimension as embeddings
                reference_indices = reference_indices.unsqueeze(2).expand_as(embeddings)

                if method == 'IntegratedGradients':
                    # Baseline set to all-zero embedding
                    attributions = explainer.attribute(
                        embeddings, 
                        baselines=reference_indices, 
                        target=labels.tolist(),
                        n_steps=20,
                        internal_batch_size=128
                    )
                elif method == 'GradientShap':
                    # Create multiple baselines
                    baselines = reference_indices
                    attributions = explainer.attribute(
                        embeddings, 
                        baselines=baselines, 
                        target=labels.tolist()
                    )
                elif method == 'DeepLift':
                    attributions = explainer.attribute(
                        embeddings, 
                        baselines=reference_indices,
                        target=labels.tolist()
                    )
                elif method == 'Saliency':
                    attributions = explainer.attribute(
                        embeddings, 
                        target=labels.tolist()
                    )
                attributions = attributions.mean(dim=-1)
                
            elif method == 'Occlusion':
                # Occlusion method doesn't need gradients, suitable for word ID input
                attributions = explainer.attribute(
                    input_ids, 
                    sliding_window_shapes=(1,), 
                    baselines=reference_indices,
                    target=labels
                )
                
            elif method == 'FeatureAblation':
                # FeatureAblation also doesn't need gradients
                attributions = explainer.attribute(
                    input_ids, 
                    baselines=reference_indices,
                    target=labels
                )
                
            elif method == 'Lime':
                # LIME also doesn't need gradients
                attributions = explainer.attribute(
                    input_ids, 
                    target=labels, 
                    baselines=reference_indices,
                    n_samples=50
                )
                
            elif method == 'KernelShap':
                # KernelShap also doesn't need gradients
                attributions = explainer.attribute(
                    input_ids, 
                    target=labels, 
                    baselines=reference_indices,
                    n_samples=50
                )

            else:
                raise ValueError(f"Unknown explanation method: {method}")

            all_attributions.append(attributions.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_texts.append(input_ids.detach().cpu())

        except Exception as e:
            print(f"Batch {batch_idx+1} explanation failed: {e}")
            continue
        
        if max_batches is not None and batch_idx + 1 >= max_batches:
            break
    
    # Concatenate all batches
    all_attributions = torch.cat(all_attributions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_texts = torch.cat(all_texts, dim=0)
    
    return all_attributions.numpy(), all_labels.numpy(), all_texts.numpy()

def load_test_data_from_main(train_path, test_path, max_len=100,vocab_path = None):
    """Directly use transformer_main.py data processing function"""
    # imdb
    X_train, y_train, X_test, y_test, vocab_size, vocab, word2idx = preprocess_data_imdb(train_path, test_path, max_len, vocab_path)
    
    #agnews
    # X_train, y_train, X_test, y_test, vocab_size, vocab, word2idx = preprocess_data_agnews(train_path, test_path, max_len)
    # Read original data for saving text
    test_df = pd.read_parquet(test_path)
    test_df['label'] = test_df['label'].astype(int)
    test_df['text'] = test_df['text'].astype(str)
    
    return X_test, y_test, test_df, vocab, word2idx

# 用法示例
if __name__ == "__main__":
    import warnings
    from warnings import simplefilter
    # Filter warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
    except ImportError:
        pass

    # Set environment variable to reduce memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    max_len = 300
    # Load data and model
    train_path = '../../../data/text_data/agnews/train-00000-of-00001.parquet'
    test_path = '../../../data/text_data/agnews/test-00000-of-00001.parquet'
    batch_size = 32  # Reduce batch size
    vocab_path = None

    
    X_test, y_test, test_df, vocab, word2idx = load_test_data_from_main(train_path, test_path, max_len=max_len,vocab_path=vocab_path)
    test_loader = create_data_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

    #transformer_agnews
    vocab_size = len(vocab)
    embed_size = 512  
    num_heads = 8
    hidden_dim = 512
    num_layers = 6
    output_size = 4  


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model_path = '../../../models/agnews/model_transformer.pth'
    wrapped_model, embedding_model, original_model = load_model(
        model_path=model_path,
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_size=output_size,
        max_len=max_len, 
        device=device
    )
    
    # Define all available explanation methods
    methods = [
        'IntegratedGradients',
        'GradientShap', 
        'DeepLift',
        'Saliency',
        'Occlusion',
        'FeatureAblation',
        'Lime',
        'KernelShap'
    ]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    PAD_IND = word2idx['<PAD>']

    # Batch explanation for all methods
    print("Starting explanation for all methods...")
    
    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running {method}...")
        print(f"{'='*50}")
        
        attributions, labels, texts = explain_batch(
            test_loader, wrapped_model, embedding_model, method=method, device=device, max_batches=None, batch_size=batch_size, PAD_IND=PAD_IND
        )
        
        print(f"{method} explanation completed: {attributions.shape}, {labels.shape}, {texts.shape}")

        # ========== Sample-level saving ========== #
        save_root = '../../../data/text_data/text_attr'
        print(f"Starting to save data for {len(attributions)} samples...")
        for i in range(len(attributions)):
            if i % 100 == 0:  # Print progress every 100 samples
                print(f"Save progress: {i}/{len(attributions)}")
            
            # Use index as sample name
            sample_name = f"{i:05d}"
            sample_dir = os.path.join(save_root, sample_name)
            saliency_dir = os.path.join(sample_dir, 'saliency')
            os.makedirs(saliency_dir, exist_ok=True)

            # Save original text
            try:
                with open(os.path.join(sample_dir, 'original.txt'), 'w', encoding='utf-8') as f:
                    f.write(test_df.iloc[i]['text'])
            except Exception as e:
                with open(os.path.join(sample_dir, 'original.txt'), 'w', encoding='utf-8') as f:
                    f.write('N/A')

            # Save input_ids
            np.save(os.path.join(sample_dir, 'input_ids.npy'), texts[i])

            # Save attribution results
            np.save(os.path.join(saliency_dir, f'{method}.npy'), attributions[i])
            
