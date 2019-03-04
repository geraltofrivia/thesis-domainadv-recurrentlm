# Major TODOs

- ~~Check if gradients are actually reversed.~~
    - ~~Check if the gradients are reversed according to specification~~
- ~~Get Domain Classification Accuracy~~
     
- Fix all hyperparams as the same as the org implementation
- Run it in default setting and replicate their results verbatim
    - Try to run their results without mixing val data during phase 2.  
    
- Convert the 2-layer lin decoder to have no activation on first, and a softmax on second (eq.4 https://arxiv.org/pdf/1704.04368.pdf) 
- Better saving stuff. Put ph2ph3 in one folder somehow, and everything.


# things which might work out

1. Using words from wiki during phase2 (implmenentation done)
2. Skipping phase1 (just train from scratch using DNN) (implementation 20min)
3. Using different optimizer than ADAM (SGD) (implementation 1min)
4. Using diff learning rate for dom clf ($lambda$ in paper) (implementation 2hr)
5. Different datasets.

How do i decide when something finally works?


## 18th Feb
~~RAN main.py with scaled loss.
Previously, the train acc (p2) rested around 28%; dann acc rested around 50% while val went to 0%
Now, both train and val go to 0, while dann rests around 55%.
What's happening? No clue!~~ [FIXED: Was a bug in code.]

~~* Loss scaling factor 0.5 -> disproportionate focus on improving both aux and normal loss (both loss decreasing).~~




## Exp Schedule

1. Run DANN code with lambda = 1.0 and see if it works as we want.
1. What happens when we run main.py without DANN?

    1. What happens when we run main.py without tied weights without DANN?
        Kinda like when we run phase2.py without tied weights.
        
----------------------------------------------------------------

# New Start

### 4th March
So we ran the code with $l\ambda$ = 0.5 and 1.0 and we get same results.
Our code does look like we reverse grads. **It should, we checked the testbench**

**Conclusion: Both ways (new implementation and old) are equal. Thus probably correct.**

 