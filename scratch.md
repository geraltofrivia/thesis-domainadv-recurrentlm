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

1. Using words from wiki during phase2
2. Skipping phase1
3. Using different optimizer than ADAM (SGD)
4. Using diff learning rate for dom clf ($lambda$ in paper)
5. Different datasets.

How do i decide when something finally works?