# GenBalance
A few months ago, I set myself the challenge of developing a new and intriguing algorithm in under 5 days. The outcome was _GenBalance_. The idea is shared by [this paper](https://arxiv.org/abs/2207.06084) -- aiming to preprocess a dataset to address both class imbalance and fairness. 

I used the same overall methodology of [GenFair](https://github.com/FedericoMz/GenFair/tree/main), from which I (dirtily) recycled most of the code. Instead of removing DN and PP, we only increase DP and PN to balance both fairness and class distribution.

Alas, while it _does_ technically work, the results fall short compared to the algorithm of Dablain et al. It's unlikely I'll revisit this project. Still, it was a fun ride!
