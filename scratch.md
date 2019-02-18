# Major TODOs

- ~~Check if gradients are actually reversed.~~
    - ~~Check if the gradients are reversed according to specification~~
- ~~Get Domain Classification Accuracy~~
     
- Fix all hyperparams as the same as the org implementation
- Run it in default setting and replicate their results verbatim
    - Try to run their results without mixing val data during phase 2.  
    
- Convert the 2-layer lin decoder to have no activation on first, and a softmax on second (eq.4 https://arxiv.org/pdf/1704.04368.pdf) 



# Major Experiments 

Or things which might work out

1. Using words from wiki during phase2 (implmenentation done)
2. Skipping phase1 (just train from scratch using DNN) (implementation 20min)
3. Using different optimizer than ADAM (SGD) (implementation 1min)
4. Using diff learning rate for dom clf ($lambda$ in paper) (implementation 2hr)
5. Different datasets.

How do i decide when something finally works?


## 18th Feb
RAN main.py with scaled loss.
Previously, the train acc (p2) rested around 28%; dann acc rested around 50% while val went to 0%
Now, both train and val go to 0, while dann rests around 55%.
What's happening? No clue!


## Detailed Exp

1. What happens when we run main.py without DANN?

    1.1 What happens when we run main.py without tied weights without DANN?
        Kinda like when we run phase2.py without tied weights.