# MoRecon

MoRecon - A tool for reconstructing missing frames in motion capture data

[バンダイナムコ研究所・データサイエンス・チャレンジ](https://athletix.run/challenges/MQe8jPDRp)の優勝解法です。


## 結果

https://user-images.githubusercontent.com/30839669/151743775-dd911609-4907-4bab-bd22-635c2babc19a.mp4

## 解法

解法解説に使用したスライドを一部掲載します。

![スライド1](https://user-images.githubusercontent.com/30839669/156153557-30d04ac6-e9fd-4dd5-be87-fdf99202eeab.PNG)

本コンペでは、数フレームに1フレームだけデータが存在するモーションキャプチャデータが与えられ、欠落しているフレームのデータを補間します。

![スライド3](https://user-images.githubusercontent.com/30839669/156153679-ace80706-ba75-49fd-a5d2-3829e64f5ee6.PNG)
![スライド4](https://user-images.githubusercontent.com/30839669/156153720-1332fbbf-5673-4f7d-ba0c-fe76aa5181f0.PNG)

訓練用ファイルは全てのフレームにデータが存在し、提出用の3ファイルはそれぞれ`5`、`15`、`45`フレームに`1`フレームのみデータが存在します。

![スライド5](https://user-images.githubusercontent.com/30839669/156153755-d07f4a78-812f-4457-94d0-a9d54e2a7321.PNG)
![スライド6](https://user-images.githubusercontent.com/30839669/156160641-d1cba0b4-5869-4915-af44-5c12df507566.PNG)

モーションは「手を振る」「走る」など、大きく分けて`7~8`種類あります。

![スライド7](https://user-images.githubusercontent.com/30839669/156153781-6a108e07-8b20-4a76-8ae3-3c21eccda3d2.PNG)

モーションの種類とデータの間隔によって、**ルールベース**手法と**機械学習**手法を使い分けています。

![スライド9](https://user-images.githubusercontent.com/30839669/156166574-8928a50b-1108-42bc-be2f-28b5cbbd32cf.PNG)

ルールベース手法は単なる**線形補間**と**2次スプライン補間**の重み付け平均です。

![スライド10](https://user-images.githubusercontent.com/30839669/156166576-779eb066-f1ea-4afa-9c80-9bcc5773accc.PNG)

機械学習手法の要点を先にまとめました。難しいことは何もしていません。

![スライド11](https://user-images.githubusercontent.com/30839669/156166579-1b3f2cc8-6922-4b3a-96a6-bee98a989f15.PNG)

入力データの欠落部分は無視しつつ、未来の情報も与えてあげます。

![スライド12](https://user-images.githubusercontent.com/30839669/156166530-ae1ab2a5-c096-4e90-8924-0b02112d0af5.PNG)
![スライド13](https://user-images.githubusercontent.com/30839669/156166534-3d26907d-abbc-4618-b23b-bccaeb9f4c60.PNG)

スケルトンデータは親関節からの**相対座標**や**角度**などとして計算することも可能ですが、本手法では元々格納されている**ワールド座標のみ**を使います。

![スライド14](https://user-images.githubusercontent.com/30839669/156166536-3179058a-d060-459f-a2b9-06ed356b1b89.PNG)

学習データは速度を変えて**データ拡張**しています。これによってかなり精度が上がります。

![スライド15](https://user-images.githubusercontent.com/30839669/156166537-57667f0a-1682-4ca1-9208-52c6798c03bc.PNG)

正規化によって特徴が潰れてしまうことを防ぐため、事前にモーションを原点に固定します。

![スライド16](https://user-images.githubusercontent.com/30839669/156166542-4b432f1b-844a-4f4c-aa5b-d7a85e685248.PNG)
![スライド17](https://user-images.githubusercontent.com/30839669/156166547-4fbb7a2d-5beb-4511-9ec3-7e0a00f45843.PNG)
![スライド18](https://user-images.githubusercontent.com/30839669/156166550-1a91d5ae-9bcd-4813-990d-372c9bdc4337.PNG)

マーカーごとに重みを最適化します。

![スライド19](https://user-images.githubusercontent.com/30839669/156166554-17dd992a-22c7-43b8-8201-03570b3948ca.PNG)
![スライド20](https://user-images.githubusercontent.com/30839669/156166557-83f4de74-fd91-4f56-97fa-553f439869f6.PNG)
![スライド21](https://user-images.githubusercontent.com/30839669/156166558-a88112c4-165d-4f79-80e6-15420dfe4473.PNG)
![スライド23](https://user-images.githubusercontent.com/30839669/156166563-0bf75951-271a-4eff-8915-134e36414d3c.PNG)

質問があれば [Twitter](https://twitter.com/yknishidate) までお願いします。

![スライド24](https://user-images.githubusercontent.com/30839669/156166571-1dd4031e-ae70-4284-b55d-bfafc31909e4.PNG)
