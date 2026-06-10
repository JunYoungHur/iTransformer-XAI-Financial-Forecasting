import os
import random
import warnings
import requests
from io import StringIO


import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

warnings.filterwarnings('ignore')


class Config:
    SEED = 42
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    TRAIN_END  = '2024-10-31'
    VAL_START  = '2024-12-15'
    VAL_END    = '2025-09-30'
    TEST_START = '2025-11-15'


    WINDOW_SIZE = 40
    HORIZON     = 21


    BATCH_SIZE   = 512
    EPOCHS       = 50
    LR           = 5e-4
    WEIGHT_DECAY = 1e-4
    PATIENCE     = 7
    GRAD_CLIP    = 1.0


    D_MODEL      = 64
    N_HEAD       = 4
    NUM_LAYERS   = 2
    DROPOUT      = 0.2
    VSN_HIDDEN   = 32


    VSN_TYPE = 'gru'


def seed_everything(seed: int = Config.SEED) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DataPipeline:


    @staticmethod
    def download_sp500(start_date: str = "2000-01-01",
                       save_path: str = None) -> pd.DataFrame:
        if save_path is None:
            save_path = str(DATA_DIR / "sp500_market_data_2000.csv")
        if os.path.exists(save_path):
            print(f"📁 Using cached price data: {save_path}")
            return pd.read_csv(save_path, index_col=0, parse_dates=True)

        print(f"📥 Downloading S&P 500 prices ({start_date} ~ now)...")

        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        table = pd.read_html(StringIO(response.text))[0]
        tickers = [t.replace('.', '-') for t in table['Symbol'].tolist()]

        data = yf.download(tickers, start=start_date,
                           group_by='ticker', auto_adjust=True, progress=True)

        df_list = []
        for ticker in tickers:
            if ticker in data.columns.levels[0]:
                temp = data[ticker].dropna(subset=['Close']).copy()
                if len(temp) > 0:
                    temp['Ticker'] = ticker
                    df_list.append(temp[['Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']])

        final_df = pd.concat(df_list)
        final_df.index.name = 'Date'
        final_df.to_csv(save_path)
        return final_df


    @staticmethod
    def build_features(df: pd.DataFrame, min_days: int = 1000) -> pd.DataFrame:
        print("🛠️  Building features (aggressive technical + sector-aware macro)...")


        valid_tickers = [t for t, g in df.groupby('Ticker')
                         if (g['Close'] > 0).sum() >= min_days]
        df = df[df['Ticker'].isin(valid_tickers)]


        expanded_dfs = []
        for ticker, group in df.groupby('Ticker'):
            group = group[group['Close'] > 0].copy()
            if len(group) < 60:
                continue

            close, high, low, volume = group['Close'], group['High'], group['Low'], group['Volume']


            for length in [7, 14, 21, 28]:
                group[f'RSI_{length}'] = ta.rsi(close, length=length)


            macd = ta.macd(close)
            if macd is not None:
                group['MACD_Ratio']      = macd.iloc[:, 0] / (close + 1e-8)
                group['MACD_Signal_Diff'] = (macd.iloc[:, 0] - macd.iloc[:, 1]) / (close + 1e-8)


            for length in [10, 20]:
                group[f'ATR_Ratio_{length}'] = ta.atr(high, low, close, length=length) / (close + 1e-8)
                bb = ta.bbands(close, length=length)
                if bb is not None:
                    group[f'BB_Percent_{length}'] = bb.iloc[:, 3]


            for length in [10, 20]:
                group[f'ROC_{length}'] = ta.roc(close, length=length)
            stoch = ta.stoch(high, low, close)
            if stoch is not None:
                group['Stoch_K'] = stoch.iloc[:, 0]
                group['Stoch_D'] = stoch.iloc[:, 1]
            group['Williams_R_14'] = ta.willr(high, low, close, length=14)


            adx = ta.adx(high, low, close, length=14)
            if adx is not None:
                group['ADX_14'] = adx.iloc[:, 0]
            group['CCI_20'] = ta.cci(high, low, close, length=20)


            obv = ta.obv(close, volume)
            if obv is not None:
                obv_ma = obv.rolling(20).mean()
                group['OBV_Ratio_20'] = (obv - obv_ma) / (obv_ma.abs() + 1e-8)
            group['MFI_14']   = ta.mfi(high, low, close, volume, length=14)
            group['CMF_20']   = ta.cmf(high, low, close, volume, length=20)
            vol_ma = volume.rolling(20).mean()
            group['Volume_Anomaly_20'] = (volume - vol_ma) / (vol_ma + 1e-8)

            expanded_dfs.append(group.dropna())

        df = pd.concat(expanded_dfs)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)


        start_date = df.index.min().strftime('%Y-%m-%d')
        end_date   = (df.index.max() + pd.Timedelta(days=2)).strftime('%Y-%m-%d')

        macro_tickers = {
            '^VIX': 'VIX', '^TNX': 'TNX_10Y', '^IRX': 'IRX_3M',
            'DX-Y.NYB': 'DXY', 'GLD': 'Gold', 'USO': 'Oil',
        }
        macro_raw = yf.download(list(macro_tickers.keys()),
                                start=start_date, end=end_date, progress=False)
        macro_close = macro_raw['Close'].rename(columns=macro_tickers).ffill()

        macro_features = pd.DataFrame(index=macro_close.index)
        for col in ['VIX', 'TNX_10Y', 'DXY', 'Gold', 'Oil']:
            macro_features[f'{col}_5d_log_ret']  = np.log(macro_close[col] / (macro_close[col].shift(5)  + 1e-8))
            macro_features[f'{col}_20d_log_ret'] = np.log(macro_close[col] / (macro_close[col].shift(20) + 1e-8))

        macro_features['Yield_Curve_Spread'] = macro_close['TNX_10Y'] - macro_close['IRX_3M']
        vix_200ma = macro_close['VIX'].rolling(200).mean()
        macro_features['VIX_vs_200MA'] = (macro_close['VIX'] - vix_200ma) / (vix_200ma + 1e-8)


        macro_features = macro_features.shift(1)
        macro_features.index = pd.to_datetime(macro_features.index, utc=True).tz_localize(None)


        sector_tickers = {
            'XLK': 'Tech', 'XLF': 'Fin',  'XLE': 'Energy', 'XLV': 'Health',
            'XLY': 'Disc', 'XLI': 'Ind',  'XLP': 'Staples','XLU': 'Util',
        }
        sector_raw = yf.download(list(sector_tickers.keys()),
                                 start=start_date, end=end_date, progress=False)
        sector_close = sector_raw['Close'].rename(columns=sector_tickers).ffill()


        spy_raw = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy = spy_raw['Close'].squeeze().ffill()
        spy.index = pd.to_datetime(spy.index, utc=True).tz_localize(None)

        spy_ret_20d     = np.log(spy / (spy.shift(20) + 1e-8)).shift(1)
        spy_200ma       = spy.rolling(200).mean()
        spy_vs_200ma    = ((spy - spy_200ma) / (spy_200ma + 1e-8)).shift(1)

        sector_features = pd.DataFrame(index=sector_close.index)
        for etf, name in sector_tickers.items():
            sec_ret_20d = np.log(sector_close[name] / (sector_close[name].shift(20) + 1e-8))
            sector_features[f'{name}_RS_20d'] = (sec_ret_20d - spy_ret_20d).shift(1)
        sector_features.index = pd.to_datetime(sector_features.index, utc=True).tz_localize(None)


        merged_dfs = []
        for ticker, group in df.groupby('Ticker'):
            group = group.copy()
            group = group.join(macro_features,  how='left')
            group = group.join(sector_features, how='left')

            stock_ret_20d = np.log(group['Close'] / (group['Close'].shift(20) + 1e-8)).shift(1)
            group['Relative_Strength_20d'] = stock_ret_20d - spy_ret_20d.reindex(group.index).ffill()
            group['Market_Regime_200MA']   = spy_vs_200ma.reindex(group.index).ffill()

            merged_dfs.append(group.dropna())

        return pd.concat(merged_dfs)


    @staticmethod
    def create_labels(df: pd.DataFrame,
                      horizon: int = 21,
                      lookback: int = 20,
                      multiplier: float = 0.5) -> pd.DataFrame:
        print("🎯 Dynamic-threshold ternary labeling...")
        df['Target_Return'] = df.groupby('Ticker')['Close'].shift(-horizon)
        df['Log_Return']    = np.log((df['Target_Return'] + 1e-8) / (df['Close'] + 1e-8))

        df['Daily_Log_Ret'] = df.groupby('Ticker')['Close'].transform(
            lambda x: np.log(x / (x.shift(1) + 1e-8))
        )
        df['Vol_Rolling']  = df.groupby('Ticker')['Daily_Log_Ret'].transform(
            lambda x: x.rolling(window=lookback).std()
        )
        df['Dynamic_Threshold'] = (df['Vol_Rolling'] * np.sqrt(horizon) * multiplier).clip(lower=0.01)

        df['Target'] = 1
        df.loc[df['Log_Return'] >  df['Dynamic_Threshold'], 'Target'] = 2
        df.loc[df['Log_Return'] < -df['Dynamic_Threshold'], 'Target'] = 0


        df = df.dropna(subset=['Log_Return'])

        return df.drop(columns=[
            'Target_Return', 'Log_Return', 'Daily_Log_Ret',
            'Vol_Rolling',  'Dynamic_Threshold',
        ])


    @staticmethod
    def list_features(df: pd.DataFrame) -> list:
        return [c for c in df.columns if c not in ['Ticker', 'Target', 'Close']]


class TimeSeriesDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: list, train_medians: pd.Series):
        self.features    = features
        self.window_size = Config.WINDOW_SIZE

        df_cleaned = df.copy()
        for col in features:
            df_cleaned[col] = df_cleaned[col].replace([np.inf, -np.inf], np.nan)


            df_cleaned[col] = (
                df_cleaned.groupby('Ticker')[col].transform(lambda s: s.ffill())
            )
            df_cleaned[col] = df_cleaned[col].fillna(train_medians[col])

        self.data_list, self.target_list, self.indices = [], [], []
        for ticker, group in df_cleaned.groupby('Ticker'):
            group = group.sort_index()
            vals    = group[features].values.astype(np.float32)
            targets = group['Target'].values.astype(np.int64)
            if len(vals) > self.window_size:
                self.data_list.append(vals)
                self.target_list.append(targets)
                t_idx = len(self.data_list) - 1
                for i in range(len(vals) - self.window_size):
                    self.indices.append((t_idx, i))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t_idx, start_i = self.indices[idx]
        window = self.data_list[t_idx][start_i : start_i + self.window_size]
        target = self.target_list[t_idx][start_i + self.window_size]
        return torch.from_numpy(window), torch.tensor(target)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta  = nn.Parameter(torch.zeros(num_features))
        self.mean  = None
        self.stdev = None

    def forward(self, x, mode: str = 'norm'):
        if mode == 'norm':
            self.mean  = x.mean(dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            x = x * self.gamma + self.beta
        elif mode == 'denorm':


            x = (x - self.beta) / self.gamma
            x = x * self.stdev + self.mean
        return x


class GatedResidualNetwork(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_out)
        self.gate    = nn.Linear(d_in, d_out)
        self.elu     = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.norm    = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.elu(self.linear1(x))
        h = self.dropout(self.linear2(h))
        g = self.sigmoid(self.gate(x)) * h


        if x.shape[-1] == g.shape[-1]:
            return self.norm(x + g)
        return self.norm(g)


class PointwiseVSN(nn.Module):
    def __init__(self, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.grn     = GatedResidualNetwork(num_features, num_features,
                                            num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.softmax(self.grn(x))
        return x * weights, weights


class NoVSN(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x):
        B, W, F_ = x.shape
        weights = torch.full((B, W, F_), 1.0 / F_, device=x.device, dtype=x.dtype)
        return x, weights


class TemporalVSN_GRU(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.gru     = nn.GRU(num_features, hidden_dim, batch_first=True)
        self.grn     = GatedResidualNetwork(hidden_dim, hidden_dim, num_features, dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        context, _ = self.gru(x)
        weights    = self.softmax(self.grn(context))
        return x * weights, weights


def build_vsn(num_features: int, vsn_type: str = Config.VSN_TYPE) -> nn.Module:
    if vsn_type == 'gru':
        return TemporalVSN_GRU(num_features, Config.VSN_HIDDEN, Config.DROPOUT)
    if vsn_type == 'pointwise':
        return PointwiseVSN(num_features, Config.DROPOUT)
    if vsn_type == 'none':
        return NoVSN(num_features)
    raise ValueError(f"Unknown VSN_TYPE: {vsn_type}")


class GatedITransformer(nn.Module):
    def __init__(self, num_features: int, num_layers: int | None = None,
                 vsn_type: str | None = None):
        super().__init__()
        num_layers = num_layers or Config.NUM_LAYERS
        vsn_type   = vsn_type   or Config.VSN_TYPE

        self.vsn_type = vsn_type
        self.revin    = RevIN(num_features)
        self.vsn      = build_vsn(num_features, vsn_type)


        self.feature_embed = nn.Linear(Config.WINDOW_SIZE, Config.D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.D_MODEL, nhead=Config.N_HEAD,
            dropout=Config.DROPOUT, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Sequential(
            nn.Linear(Config.D_MODEL, Config.D_MODEL // 2),
            nn.ELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.D_MODEL // 2, 3),
        )

    def forward(self, x, return_weights: bool = False):

        x = self.revin(x, mode='norm')


        x, vsn_weights = self.vsn(x)


        x = x.transpose(1, 2)
        x = self.feature_embed(x)


        x = self.transformer(x)


        x = x.mean(dim=1)


        mean_vsn_weights = vsn_weights.mean(dim=1)

        return self.fc_out(x), mean_vsn_weights


class Trainer:
    def __init__(self, model: nn.Module, class_weights: np.ndarray | None = None):
        self.model = model.to(Config.DEVICE)

        if class_weights is not None:
            cw = torch.FloatTensor(class_weights).to(Config.DEVICE)
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3,
        )

    def _run_epoch(self, loader: DataLoader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss, correct, total = 0.0, 0, 0
        all_f_imps = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                if train:
                    self.optimizer.zero_grad()
                logits, f_imp = self.model(x)
                loss = self.criterion(logits, y)
                if train:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=Config.GRAD_CLIP)
                    self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total   += y.size(0)
                correct += (predicted == y).sum().item()
                if not train:
                    all_f_imps.append(f_imp.cpu())

        avg_loss = total_loss / max(1, len(loader))
        acc      = correct / max(1, total)

        if not train:
            imps = torch.cat(all_f_imps, dim=0)
            return avg_loss, acc, imps.mean(dim=0).numpy(), imps.std(dim=0).numpy()
        return avg_loss, acc

    def fit(self, train_loader, val_loader, features, ckpt_path='best_model.pth'):
        print(f"\n🔥 GatedITransformer training (device: {Config.DEVICE})")
        best_val_loss   = float('inf')
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            t_loss, t_acc                 = self._run_epoch(train_loader, train=True)
            v_loss, v_acc, v_imp, v_imp_std = self._run_epoch(val_loader,   train=False)
            self.scheduler.step(v_loss)

            print(f"Epoch [{epoch+1:02d}] "
                  f"T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | V_Acc {v_acc:.2%}")

            top3 = v_imp.argsort()[-3:][::-1]
            print("   🔝 Top-3 VSN features: "
                  + ", ".join([f"{features[i]}({v_imp[i]:.3f})" for i in top3]))

            if v_loss < best_val_loss:
                best_val_loss   = v_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"🛑 Early stopping at epoch {epoch+1}")
                    break

    def evaluate(self, loader, tag='Test'):
        loss, acc, imp, imp_std = self._run_epoch(loader, train=False)
        print(f"📊 {tag}: loss={loss:.4f}, acc={acc:.2%}")
        return loss, acc, imp, imp_std


def run_diagnostics(model, loader, run_id: str, save_dir: Path):
    print("\n" + "=" * 50)
    print(f"🩺 Diagnostics ({run_id})")

    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for x, y in loader:
            logits, _ = model(x.to(Config.DEVICE))
            probs     = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))
            all_targets.append(y.numpy())

    all_probs   = np.concatenate(all_probs)
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)


    classes = ['down', 'flat', 'up']
    pred_dist = {n: float((all_preds   == c).mean()) for c, n in enumerate(classes)}
    true_dist = {n: float((all_targets == c).mean()) for c, n in enumerate(classes)}
    mean_prob = {n: float(all_probs[:, c].mean())    for c, n in enumerate(classes)}

    print("Prediction distribution:")
    for n, v in pred_dist.items(): print(f"  {n}: {v:.2%}")
    print("True distribution (val):")
    for n, v in true_dist.items(): print(f"  {n}: {v:.2%}")


    confusion = {}
    print("\nConfusion (rows=true, cols=pred):")
    for t, tn in enumerate(classes):
        row = [int(((all_targets == t) & (all_preds == p)).sum()) for p in range(3)]
        confusion[tn] = {p_name: row[p] for p, p_name in enumerate(classes)}
        print(f"  true_{tn:<4}: " + "  ".join(f"{v:>8}" for v in row))


    directional_mask = all_preds != 1
    conf      = all_probs.max(axis=1)[directional_mask]
    correct   = (all_preds[directional_mask] == all_targets[directional_mask])
    inv_preds = np.where(all_preds == 0, 2, np.where(all_preds == 2, 0, 1))
    inv_correct = (inv_preds[directional_mask] == all_targets[directional_mask])

    confidence_buckets = {}
    print("\nAccuracy by confidence bucket (directional only):")
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        bucket = conf >= thresh
        n      = int(bucket.sum())
        if n > 0:
            acc     = float(correct[bucket].mean())
            inv_acc = float(inv_correct[bucket].mean())
        else:
            acc, inv_acc = 0.0, 0.0
        confidence_buckets[f">={thresh}"] = {'n': n, 'acc': acc, 'inverted_acc': inv_acc}
        print(f"  conf >= {thresh}: n={n:>6}, acc={acc:.2%}, inverted_acc={inv_acc:.2%}")


    diag = {
        'run_id':              run_id,
        'prediction_dist':     pred_dist,
        'true_dist_val':       true_dist,
        'mean_softmax':        mean_prob,
        'confusion_matrix':    confusion,
        'confidence_buckets':  confidence_buckets,
        'overall_accuracy':    float((all_preds == all_targets).mean()),
        'overall_inverted_acc': float((inv_preds == all_targets).mean()),
    }
    out = save_dir / f"diagnostics_{run_id}.json"
    out.write_text(json.dumps(diag, indent=2))
    print(f"\n💾 Diagnostics → {out}")
    return diag


class Analyzer:
    @staticmethod
    def report_vsn_importance(vsn_importance: np.ndarray,
                              features: list, topk: int = 10) -> pd.Series:
        ranked = pd.Series(vsn_importance, index=features).sort_values(ascending=False)
        print(f"🎯 VSN top-{topk} features (averaged over eval set):")
        for name, val in ranked.head(topk).items():
            print(f"     {name:<30s}  {val:.4f}")
        return ranked

    @staticmethod
    def plot_feature_interaction(model, loader, features,
                                 ckpt_path='best_model.pth',
                                 save_path=None):
        print("🎨 Post-VSN feature interaction map...")
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True))
        model.eval()

        gated = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(Config.DEVICE)
                x_norm        = model.revin(x, mode='norm')
                x_gated, _    = model.vsn(x_norm)
                gated.append(x_gated.mean(dim=1).cpu().numpy())

        mat       = np.concatenate(gated, axis=0)
        corr_mat  = pd.DataFrame(mat, columns=features).corr(method='spearman')

        plt.figure(figsize=(max(10, len(features) * 0.35),
                            max(8, len(features) * 0.3)))
        sns.heatmap(corr_mat, cmap='coolwarm', center=0,
                    xticklabels=features, yticklabels=features)
        plt.title("Feature Interaction Map (Post-VSN Spearman Correlation)")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"   💾 Saved → {save_path}")
        plt.show()
        return corr_mat

    @staticmethod
    def track_attention_dynamics(model, df, ticker, features,
                                 ckpt_path='best_model.pth'):
        print(f"🔍 [{ticker}] VSN attention over time...")
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True))
        model.eval()

        stock_df = df[df['Ticker'] == ticker].sort_index()
        dates, records = [], []
        values = stock_df[features].values.astype(np.float32)

        with torch.no_grad():
            for i in range(len(stock_df) - Config.WINDOW_SIZE):
                window    = values[i : i + Config.WINDOW_SIZE]
                x_tensor  = torch.tensor(window).unsqueeze(0).to(Config.DEVICE)
                _, w      = model(x_tensor)
                dates.append(stock_df.index[i + Config.WINDOW_SIZE - 1])
                records.append(w.cpu().numpy()[0])

        return pd.DataFrame(records, columns=features, index=dates)

    @staticmethod
    def analyze_confidence(model, loader, top_k_ratio=0.10,
                           ckpt_path='best_model.pth'):
        model.load_state_dict(torch.load(ckpt_path, map_location=Config.DEVICE, weights_only=True))
        model.eval()
        probs, preds, targets = [], [], []

        with torch.no_grad():
            for x, y in loader:
                logits, _ = model(x.to(Config.DEVICE))
                p, pr     = torch.max(F.softmax(logits, dim=1), 1)
                probs.extend(p.cpu().numpy())
                preds.extend(pr.cpu().numpy())
                targets.extend(y.numpy())

        df_res    = pd.DataFrame({'Confidence': probs, 'Prediction': preds, 'Target': targets})
        df_action = df_res[df_res['Prediction'] != 1].sort_values('Confidence', ascending=False)
        top_k     = df_action.head(max(1, int(len(df_action) * top_k_ratio)))


        topk_acc   = float((top_k['Prediction'] == top_k['Target']).mean()) if len(top_k) else 0.0
        long_ratio = float((top_k['Prediction'] == 2).mean())               if len(top_k) else 0.0

        print(f"🚀 Top-{int(top_k_ratio*100)}% directional bets accuracy: {topk_acc:.2%}")
        print(f"   Long ratio among bets: {long_ratio:.2%}")


        return {
            'top_k_ratio':           top_k_ratio,
            'top_k_n':               int(len(top_k)),
            'top_k_accuracy':        topk_acc,
            'top_k_long_ratio':      long_ratio,
            'directional_pool_size': int(len(df_action)),
        }


def main(weight_mode: str = 'none', vsn_type: str | None = None):
    seed_everything()


    if vsn_type is not None:
        Config.VSN_TYPE = vsn_type

    FINAL_DATA_PATH = str(DATA_DIR / "sp500_final_processed.parquet")

    print("🚀 Pipeline start...")


    if not os.path.exists(FINAL_DATA_PATH):
        raw_df      = DataPipeline.download_sp500()
        featured_df = DataPipeline.build_features(raw_df)
        labeled_df  = DataPipeline.create_labels(featured_df, horizon=Config.HORIZON)
        labeled_df.to_parquet(FINAL_DATA_PATH)
        print(f"✅ Data engineered and cached: {FINAL_DATA_PATH}")
    else:
        print(f"📁 Cached data found: {FINAL_DATA_PATH}")
        labeled_df = pd.read_parquet(FINAL_DATA_PATH)


    features = DataPipeline.list_features(labeled_df)
    print(f"   Feature count handed to model: {len(features)}")


    train_df = labeled_df[labeled_df.index <= Config.TRAIN_END]
    val_df   = labeled_df[(labeled_df.index >= Config.VAL_START) &
                          (labeled_df.index <= Config.VAL_END)]
    test_df  = labeled_df[labeled_df.index >= Config.TEST_START]

    print(f"   train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")


    train_medians = train_df[features].replace([np.inf, -np.inf], np.nan).median()


    WEIGHT_MODE = weight_mode

    counts = train_df['Target'].value_counts().sort_index().values

    if WEIGHT_MODE == 'inverse':
        weights = 1.0 / (counts + 1e-8)
        weights = weights / weights.sum() * 3
    elif WEIGHT_MODE == 'sqrt':
        weights = 1.0 / np.sqrt(counts + 1e-8)
        weights = weights / weights.sum() * 3
    elif WEIGHT_MODE == 'none':
        weights = None
    else:
        raise ValueError(f"Unknown WEIGHT_MODE: {WEIGHT_MODE}")

    print(f"⚖️  Class weights ({WEIGHT_MODE}): {weights}")


    RUN_ID    = f"{Config.VSN_TYPE}_h{Config.HORIZON}_{WEIGHT_MODE}"
    RESULTS   = RESULTS_DIR
    FIGS      = RESULTS / "figures"
    RESULTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)
    CKPT_PATH = f"best_model_{RUN_ID}.pth"
    print(f"🏷️  RUN_ID: {RUN_ID}")
    print(f"📂 results dir: {RESULTS.resolve()}")


    train_ds = TimeSeriesDataset(train_df, features, train_medians)
    val_ds   = TimeSeriesDataset(val_df,   features, train_medians)
    test_ds  = TimeSeriesDataset(test_df,  features, train_medians)

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)


    print(f"   VSN type: {Config.VSN_TYPE}")
    model   = GatedITransformer(num_features=len(features))
    trainer = Trainer(model, class_weights=weights)

    trainer.fit(train_loader, val_loader, features, ckpt_path=CKPT_PATH)


    print("\n" + "=" * 50)
    print("📊 Final evaluation on held-out TEST set")
    model.load_state_dict(torch.load(CKPT_PATH,
                                     map_location=Config.DEVICE, weights_only=True))
    test_loss, test_acc, test_imp, _ = trainer.evaluate(test_loader, tag='TEST')


    print("\n" + "=" * 50)
    print("🔬 Post-hoc analysis (XAI)")


    confidence_metrics = Analyzer.analyze_confidence(
        model, val_loader, top_k_ratio=0.1, ckpt_path=CKPT_PATH
    )

    vsn_importance = Analyzer.report_vsn_importance(test_imp, features, topk=10)


    Analyzer.plot_feature_interaction(
        model, val_loader, features,
        ckpt_path=CKPT_PATH,
        save_path=FIGS / f"feature_interaction_{RUN_ID}.png",
    )

    sample_ticker = labeled_df['Ticker'].unique()[0]
    attn_df = Analyzer.track_attention_dynamics(
        model, labeled_df, sample_ticker, features, ckpt_path=CKPT_PATH
    )
    print(f"\n📈 [{sample_ticker}] VSN weight history (tail):")
    print(attn_df.tail(5))


    attn_df.to_csv(RESULTS / f"attention_timeseries_{RUN_ID}_{sample_ticker}.csv")

    fig, ax = plt.subplots(figsize=(12, 5))
    attn_df.tail(100).plot(
        ax=ax,
        title=f"Temporal Dynamics of VSN Weights: {sample_ticker} ({RUN_ID})",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(FIGS / f"attention_dynamics_{RUN_ID}_{sample_ticker}.png",
                dpi=120, bbox_inches='tight')
    print(f"   💾 Saved → {FIGS / f'attention_dynamics_{RUN_ID}_{sample_ticker}.png'}")
    plt.show()


    diagnostics = run_diagnostics(model, val_loader, run_id=RUN_ID, save_dir=RESULTS)


    summary = {
        'run_id':    RUN_ID,
        'config': {
            'vsn_type':     Config.VSN_TYPE,
            'horizon':      Config.HORIZON,
            'window_size':  Config.WINDOW_SIZE,
            'weight_mode':  WEIGHT_MODE,
            'class_weights': (None if weights is None
                              else [float(w) for w in weights]),
            'batch_size':   Config.BATCH_SIZE,
            'lr':           Config.LR,
            'd_model':      Config.D_MODEL,
            'n_head':       Config.N_HEAD,
            'num_layers':   Config.NUM_LAYERS,
            'dropout':      Config.DROPOUT,
        },
        'data': {
            'feature_count': len(features),
            'train_rows':    int(len(train_df)),
            'val_rows':      int(len(val_df)),
            'test_rows':     int(len(test_df)),
        },
        'test_metrics': {
            'test_loss':     float(test_loss),
            'test_accuracy': float(test_acc),
        },
        'confidence_bets':   confidence_metrics,
        'vsn_top10':         {k: float(v) for k, v in
                              vsn_importance.head(10).items()},
        'diagnostics_file':  f"diagnostics_{RUN_ID}.json",
        'checkpoint_file':   CKPT_PATH,
        'figures': {
            'feature_interaction': f"figures/feature_interaction_{RUN_ID}.png",
            'attention_dynamics':  f"figures/attention_dynamics_{RUN_ID}_{sample_ticker}.png",
        },
    }
    summary_path = RESULTS / f"summary_{RUN_ID}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n📦 Run summary → {summary_path}")
    print("✅ All artifacts archived. Ready for README write-up.")


if __name__ == "__main__":
    main(weight_mode='none', vsn_type='gru')

