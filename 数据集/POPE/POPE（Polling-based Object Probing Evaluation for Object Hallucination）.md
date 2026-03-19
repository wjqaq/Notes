![](assets/POPE（Polling-based%20Object%20Probing%20Evaluation%20for%20Object%20Hallucination）/file-20260319143437785.png)
- 人类注释 + SEEM（等工具）分段后自动注释，然后进行负采样，得到（图片）真实对象和不存在对象，组合后提问。
- 方法包括：随机/流行/对抗。  这里流行应该是统计出来的幻觉错误率高的词：
![](assets/POPE（Polling-based%20Object%20Probing%20Evaluation%20for%20Object%20Hallucination）/file-20260319143728438.png)
- a：常见物体的幻觉出次数频率前十的