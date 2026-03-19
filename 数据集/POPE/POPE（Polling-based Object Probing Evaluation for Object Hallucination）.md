![](assets/POPE（Polling-based%20Object%20Probing%20Evaluation%20for%20Object%20Hallucination）/file-20260319143437785.png)
- 人类注释 + SEEM（等工具）分段后自动注释，然后进行负采样，得到（图片）真实对象和不存在对象，组合后提问。
- 方法包括：随机/流行/对抗。  这里流行应该是统计出来的幻觉错误率高的词：
![](assets/POPE（Polling-based%20Object%20Probing%20Evaluation%20for%20Object%20Hallucination）/file-20260319143728438.png)
- a：幻觉次数出现的频率前十的物体；
- b：幻觉次数前十的物体和“dining table”同时出现的频率；
