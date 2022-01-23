# search
Simple embedding based search on wikipedia.


```
xnli
           dev contradiction  dev entailment  test contradiction  test entailment
name                                                                             
bepo             1343.969880      707.027711         1346.088623       780.717964
minilm            468.398795       40.008434          520.622754        39.492814
mpnet             681.584337       22.162651          668.625749        36.349102
distiluse         147.207229       45.138554          170.205389        54.616168

sts2015
           newswire pearsonr  newswire spearmanr  wikipedia pearsonr  wikipedia spearmanr
name                                                                                     
bepo                0.521605            0.474683            0.705852             0.589607
minilm              0.658118            0.613012            0.741183             0.630976
mpnet               0.692602            0.653156            0.772762             0.669779
distiluse           0.685643            0.645312            0.707900             0.603374
```

