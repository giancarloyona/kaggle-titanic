# Titanic: Machine Learning from Disaster

## About the challenge

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy.

More info [here](https://www.kaggle.com/c/titanic).

---

## Solving the challenge

As said before, the goal is to predict which group of passengers is more likely to survive the sinking. We're given two ```.csv``` files (one containg the training and the other containg the test dataset) to develop the predicitive model.

### How it was done

For this challenge, I've used the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) approach to try to develop the eventual solution. I won't detail every single step of it for now, but I'll try - at least - to describe the the models used in this challenge.

### ML techniques used

The first model I created used a Decision Tree Regression, but I changed it to a Decision Tree Classifier. I believe  that this model in particular is still very ineffective, so I'll be returning to it frequently. Also, I think I'll develop a couple more models, with ML techniques yet to be chosen.

---

## And the final results are...

| Submission | Results       | ML Technique              |
| :--------- | :-----------: | :-----------------------: |
| #1         | *0.70334*     | Decision Tree Classifier  |
| #2         | ***0.76555*** | Random Forest Classifier  |
| #3         | *0.75598*     | Random Forest Classifier* |

For my surprise, the results turned out to be a little better than what I was expecting, but there are still a lot to do.

The next goal is to improve it; Let's say, what about trying to achieve a score of *80%*?

PS: The third model used the same technique as the second one, but I changed the encoding of a variable (OneHotEncoded instead of Ordinal)

---

## Additional notes

* v0.1 -> created the first model (Decision Tree Regression)
* v0.1.1 -> changed to Decision Tree Classifier
* v0.1.2 -> generated first results
* v0.1.2.1 -> added the results for the Random Forest Classifier