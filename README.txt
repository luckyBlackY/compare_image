＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

今回は画質比較ができる簡易的なRankNetモデルを自作したものを紹介。


そもそもRankNetとは？

RankNetは、ランキング学習（Learning to Rank）を目的としたニューラルネットワークの一種。


RankNetと通常のニューラルネットワークの違い

RankNetは、ランキング学習を行うために設計されたニューラルネットワークだが、通常のニューラルネットワークとは異なる点がいくつかある。

1. 学習対象の違い

•通常のニューラルネットワークは、入力に対して特定のクラスを予測する分類問題に使用される。たとえば、画像分類では、与えられた画像が「1」か「2」かを予測することが目標になる。
•RankNetでは、相対的な優劣を予測する。入力データがペアになっており、それらの間でどちらが「より良い」か、つまりどちらが「高いスコア」を持つかを学習する。画像の優劣を予測することができたり、検索エンジンでのランキングやレコメンデーションに応用される。

(学習対象の違いの画像)

2. 損失関数の違い

•通常のニューラルネットワークでは、クロスエントロピー損失関数などがよく使われる。これは、予測クラスと実際のクラスとの差を計算し、分類精度を高めるための損失を計算するものだ。
•RankNetでは、2つの入力データ間のスコア差をもとに、シグモイド型の損失関数を使用する。この損失関数は、2つのスコアの差に基づき、どちらが優れているかの確率を計算する。これにより、ランキングの精度が向上する。

(損失関数の違いの画像)

通常のニューラルネットワークがNクラス分類できるのと同様、RankNetで様々なペアの優劣を推論することで全体で順位をつけることができる(参考：RankNetを用いてMNISTに順位付けをする https://qiita.com/kzkadc/items/c358338f0d8bd764f514 )


データセット作成

今回の目的を改めて述べると、『画質比較ができるモデルを作成すること。』
そのためにCIFAR-10画像を使用。元画像と元画像に微小のノイズを足した画像を比較する。(※CIFAR-10は32x32pxというとても小さな画像なので、両方とも画質が悪く見えるが、そこはお許しを)
以下の二つの例はどちらも左が元画像、右がノイズを足した画像である(ノイズの標準偏差=0.01の画像、詳細は補足のコードを参照)。
(example1, example2の画像)
よく見ると、道路や車にノイズがあるのがわかるが、かなり見つけづらい。

これらをペアとしてデータセットを作成(ノイズの標準偏差=0.001とし、50000件のペアを作成)
また、先ほどの例では左が元画像だったが、結果が表示されるときに全て左が良いと推論されているのを見るのはつまらないので、データセットを作成する際は50%の確率で右が元画像になるようにした。


結果

学習コードは補足を参照。
30エポックで学習した。各エポックでの学習データでの損失と正解率は下記の通りである。
Epoch [1/30], Loss: 0.1619, Accuracy: 0.9355
Epoch [2/30], Loss: 0.0475, Accuracy: 0.9818
Epoch [3/30], Loss: 0.0370, Accuracy: 0.9858
Epoch [4/30], Loss: 0.0321, Accuracy: 0.9884
Epoch [5/30], Loss: 0.0273, Accuracy: 0.9896
Epoch [6/30], Loss: 0.0223, Accuracy: 0.9917
Epoch [7/30], Loss: 0.0223, Accuracy: 0.9917
Epoch [8/30], Loss: 0.0209, Accuracy: 0.9925
Epoch [9/30], Loss: 0.0204, Accuracy: 0.9927
Epoch [10/30], Loss: 0.0183, Accuracy: 0.9934
Epoch [11/30], Loss: 0.0170, Accuracy: 0.9932
Epoch [12/30], Loss: 0.0161, Accuracy: 0.9938
Epoch [13/30], Loss: 0.0157, Accuracy: 0.9942
Epoch [14/30], Loss: 0.0169, Accuracy: 0.9936
Epoch [15/30], Loss: 0.0132, Accuracy: 0.9950
Epoch [16/30], Loss: 0.0152, Accuracy: 0.9946
Epoch [17/30], Loss: 0.0146, Accuracy: 0.9949
Epoch [18/30], Loss: 0.0132, Accuracy: 0.9949
Epoch [19/30], Loss: 0.0108, Accuracy: 0.9962
Epoch [20/30], Loss: 0.0131, Accuracy: 0.9952
Epoch [21/30], Loss: 0.0150, Accuracy: 0.9949
Epoch [22/30], Loss: 0.0139, Accuracy: 0.9950
Epoch [23/30], Loss: 0.0120, Accuracy: 0.9956
Epoch [24/30], Loss: 0.0113, Accuracy: 0.9958
Epoch [25/30], Loss: 0.0121, Accuracy: 0.9957
Epoch [26/30], Loss: 0.0117, Accuracy: 0.9958
Epoch [27/30], Loss: 0.0111, Accuracy: 0.9962
Epoch [28/30], Loss: 0.0090, Accuracy: 0.9968
Epoch [29/30], Loss: 0.0090, Accuracy: 0.9965
Epoch [30/30], Loss: 0.0103, Accuracy: 0.9961

1エポック目の時点ですでに高い正答率を出し、28エポック目で99.7%近くの正解率になった(学習にかかった時間は3分！短くてありがたい！)。
その時のモデルを使用して、テストデータを使用して精度を確認したところ、ノイズの標準偏差を0.001とした時は99.45%と非常に高い精度となっていた(標準偏差を0.01にしたら100%だった)。

いくつかうまく優劣をつけられた画像を紹介する(今回は簡易的な実験のため、予測ミスしたデータの分析は省略)。
下の二つはノイズの標準偏差が0.01の時の結果である。
画像の下に書かれている数字はモデルが出力したスコアである。
(result_1.png)
img1 : [-27733.006], img2 : [-28623.139]
(result_2.png)
img1 : [-28990.162], img2 : [-28085.668]
どちらの例においてもスコアの差は約900である。

続いて、ノイズの標準偏差が0.001の時はどうか。
(result_3.png)
img1 : [-26810.15], img2 : [-26797.92]
(result_4.png)
img1 : [-29601.906], img2 : [-29608.998]
人間の目じゃ違いがわからないくらいとても似た画像となっている(少なくとも黒木はどこに違いがあるかわからない)。しかし、モデルは正しい優劣をつけることに成功している。ただ、違いがほぼ無いので、スコアの差は約10とすごく小さくなった(なお、シグモイド関数に入れて確率を求めるとほぼ1ではある)。


結論
自作の簡易的なモデル構造でも画質の比較をするモデルを作成することに成功。


今後
自己学習についてはこれで終了だが、実業務のことを考えると課題がある。
実際の写真を扱う場合、画像の大きさは今回のCIFAR-10と比べて圧倒的にでかい。そのため、自作のモデル構造がそのまま通用できるかはかなり怪しい。
層をもっと深くしたり構造を考え直すか、ResNetなどの既存モデルの出力層を変更するかなどを検討した方がいい。


補足：
googleのColab notebookで実行(T4 GPUを使用)
コードを紹介。
```
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
```

```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用デバイス: {device}')
```

```
class ImageQualityDataset(Dataset):
    def __init__(self, dataset, noise_std=0.001):
        self.noise_std = noise_std
        self.dataset = dataset
        self.data = self.generate_data()
        
    def generate_data(self):
        data = []
        for img, _ in self.dataset:
            img1 = img.clone()  # 画像データのクローンを作成
            img2 = img1 + torch.randn_like(img1) * self.noise_std  # ノイズを追加
            img2 = torch.clamp(img2, 0, 1)  # 値を[0,1]にクリップ
            
            # ランダムで左右を入れ替える(0:右が良い, 1:左が良い)
            if np.random.rand() > 0.5:
                data.append((img1, img2, torch.tensor([1.0])))
            else:
                data.append((img2, img1, torch.tensor([0.0])))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img1, img2, label = self.data[idx]
        return img1.to(device), img2.to(device), label.to(device)
```

```
class RankNet(nn.Module):
    def __init__(self):
        super(RankNet, self).__init__()
        # 畳み込み層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全結合層
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # 畳み込みとプーリングの繰り返し
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # 平坦化
        x = x.view(x.size(0), -1)
        
        # 全結合層を通過
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

```
def get_dataloader(batch_size, noise_std, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    quality_dataset = ImageQualityDataset(dataset, noise_std=noise_std)
    dataloader = DataLoader(quality_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
```

```
def ranknet_loss(output1, output2, target):
    prob = torch.sigmoid(output1 - output2).squeeze()
    return nn.BCELoss()(prob, target.squeeze())
```

```
def train_ranknet(model, dataloader, optimizer, epochs):
    model.train()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for img1, img2, labels in dataloader:
            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)
            loss = ranknet_loss(output1, output2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 予測の計算
            predictions = (output1 > output2).float().squeeze()
            correct_predictions += (predictions == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)

        # エポックごとの損失と精度の計算
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_predictions

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
        # 最良のモデルを保存
        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            best_epoch = epoch + 1
            # モデルの保存
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best model saved at epoch {best_epoch} with accuracy {best_val_acc:.4f}")
```

```
def run_experiment():
    # ハイパーパラメータ
    batch_size = 64
    epochs = 30
    learning_rate = 0.001
    noise_std = 0.001  # ノイズの標準偏差を増加
    
    # データローダー取得
    train_loader = get_dataloader(batch_size, noise_std, train=True)
    
    # RankNetモデルと最適化関数の定義
    model = RankNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # モデルのトレーニング
    train_ranknet(model, train_loader, optimizer, epochs)
```

```
if __name__ == "__main__":
    run_experiment()
```

```
def run_inference(ranknet, dataloader):
    print("\n推論開始...")
    ranknet.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for img1, img2, labels in dataloader:
            predictions = []
            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)

            output1 = ranknet(img1)
            output2 = ranknet(img2)
            for i in range(len(output1)):
                print(f"img1 : {output1[i].cpu().numpy()}, img2 : {output2[i].cpu().numpy()}")
            diff = output1 - output2
            preds = (diff > 0).float().squeeze()
            predictions.extend(preds.cpu().numpy())

            # 画像表示(見たい時にコメントアウトを外す)
            for i in range(len(predictions)):
                actual_label = "img1 > img2" if labels[i].item() == 1.0 else "img1 <= img2"
                predicted_label = "img1 > img2" if predictions[i] == 1.0 else "img1 <= img2"
                print(f"Image Pair {i + 1}: Predicted: {predicted_label}, Actual: {actual_label}")

                # 画像の表示
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                # img1
                axs[0].imshow(np.transpose(img1[i].cpu().numpy(), (1, 2, 0)))
                axs[0].set_title('Image 1')
                axs[0].axis('off')

                # img2
                axs[1].imshow(np.transpose(img2[i].cpu().numpy(), (1, 2, 0)))
                axs[1].set_title('Image 2')
                axs[1].axis('off')

                plt.suptitle(f"Predicted: {predicted_label}\nActual: {actual_label}")
                plt.show()

            break  # 最初のバッチのみ表示

            # 予測の正誤を判定してカウント
            correct_predictions += (preds == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)

        # 精度の計算
        accuracy = correct_predictions / total_predictions
        print(f"Test Accuracy: {accuracy:.4f}")
```

```
if __name__ == "__main__":
    # 推論用データローダーを取得
    test_loader = get_dataloader(batch_size=16, noise_std=0.001, train=False)

    # モデルのロード
    ranknet = RankNet().to(device)
    ranknet.load_state_dict(torch.load("best_model.pth"))

    # 推論の実行
    run_inference(ranknet, test_loader)
```

余談：
先ほどの補足ではcolab notebookを使用したと言った。
実は、最初はDockerの勉強のために私物ノートPC内でコンテナを立てて、そこで作業をしていた。
そのため、学習はCPUで行っていた。

しかし、やはりCPUでの学習は時間がかかる。。
下は本当は10エポックでやろうと思った時の結果であるが、10m50sで3エポックしか進まなかった。
途中で述べたようにGPU使用時は3minで30エポックを終了したので、CPUではかなり時間がかかっていることがわかる。
Epoch [1/10], Loss: 0.1884, Accuracy: 0.9262
Epoch [2/10], Loss: 0.0568, Accuracy: 0.9784
Epoch [3/10], Loss: 0.0368, Accuracy: 0.9861

あと、CPU使用率もやばいしパソコンが熱くなりすぎたので中断。
(CPU画像)

なので、私物ノートPCにダメージがいかないように、今後もありがたくcolab notebookを使わせていただこうと思った(dockerの勉強は別で行う)。