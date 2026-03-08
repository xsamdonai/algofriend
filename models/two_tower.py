import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# -------------------------------------------------------------
# Deep Neural Network Encoders 🧠
# -------------------------------------------------------------

class UserEncoder(nn.Module):
    """
    Encodes user demographic and behavioral data into a dense vector space. 🧑‍💻
    """
    def __init__(self, num_users: int = 10000, user_feature_dim: int = 20, embedding_dim: int = 128):
        super(UserEncoder, self).__init__()
        # Embedding for user ID
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Dense layers for processing continuous/categorical user features (e.g., age, geo)
        self.feature_net = nn.Sequential(
            nn.Linear(user_feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        
        # Fusion and output projection
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor) -> torch.Tensor:
        id_emb = self.user_embedding(user_ids)
        feat_emb = self.feature_net(user_features)
        
        fused = torch.cat([id_emb, feat_emb], dim=1)
        out = self.fc(fused)
        
        # L2 Normalize for cosine similarity operations 📏
        return F.normalize(out, p=2, dim=1)


class ItemEncoder(nn.Module):
    """
    Encodes item text/metadata (tags, categories) into a dense vector space. 📦
    """
    def __init__(self, num_items: int = 50000, item_feature_dim: int = 50, embedding_dim: int = 128):
        super(ItemEncoder, self).__init__()
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.feature_net = nn.Sequential(
            nn.Linear(item_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        id_emb = self.item_embedding(item_ids)
        feat_emb = self.feature_net(item_features)
        
        fused = torch.cat([id_emb, feat_emb], dim=1)
        out = self.fc(fused)
        
        return F.normalize(out, p=2, dim=1)


# -------------------------------------------------------------
# Main Two-Tower Model Wrapper 🗼🗼
# -------------------------------------------------------------

class TwoTowerRecommendationModel(nn.Module):
    """
    Dual-encoder architecture where User and Item towers output vectors
    into the same dense semantic space. 🌌
    Distance (dot product) between vectors correlates to interaction likelihood.
    """
    def __init__(self, embedding_dim: int = 128):
        super(TwoTowerRecommendationModel, self).__init__()
        self.user_encoder = UserEncoder(embedding_dim=embedding_dim)
        self.item_encoder = ItemEncoder(embedding_dim=embedding_dim)

    def forward(self, user_ids: torch.Tensor, user_features: torch.Tensor, item_ids: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        """
        Usually used during training to compute the similarity score
        between a batch of users and their interacted items. 🎯
        """
        user_embs = self.user_encoder(user_ids, user_features)
        item_embs = self.item_encoder(item_ids, item_features)
        
        # Cosine similarity score (since outputs are L2 normalized, dot product == cosine sim)
        scores = torch.sum(user_embs * item_embs, dim=1)
        return scores

    # --- Inference Helpers ---

    def encode_user(self, user_id: str) -> List[float]:
        """
        MOCKS retrieving online features from Feast, running the NN forward pass,
        and returning the list embedding for FAISS index search. 🔎
        """
        self.eval()
        with torch.no_grad():
            uid_tensor = torch.tensor([hash(user_id) % 10000])
            feat_tensor = torch.randn(1, 20) # Mock Feast features
            
            emb = self.user_encoder(uid_tensor, feat_tensor)
            return emb[0].tolist()

    def encode_item(self, item_id: str) -> List[float]:
        self.eval()
        with torch.no_grad():
            iid_tensor = torch.tensor([hash(item_id) % 50000])
            feat_tensor = torch.randn(1, 50)
            
            emb = self.item_encoder(iid_tensor, feat_tensor)
            return emb[0].tolist()


# -------------------------------------------------------------
# Contrastive Loss (InfoNCE) Concept for Training ⚖️
# -------------------------------------------------------------
class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, user_embs: torch.Tensor, item_embs: torch.Tensor) -> torch.Tensor:
        """
        Pulls positive pairs together and pushes in-batch negative pairs apart. 🧲
        """
        # Compute similarity matrix [Batch, Batch]
        logits = torch.matmul(user_embs, item_embs.transpose(0, 1)) / self.temperature
        
        # Labels are the diagonal (user i interacted with item i)
        labels = torch.arange(user_embs.size(0)).to(user_embs.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss

# Instantiate a global instance for the API serving layer to use 🌐
two_tower_model = TwoTowerRecommendationModel()
