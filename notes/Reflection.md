# Reflection
## 1. Introduction

After a handful of years since the introduction of AI collaboration through Large Language models (LLMs), it has become clear that not all use of these machines is equal. It has become a trend of allocating all of your mental processing onto LLMs, perhaps, because it promises a solution to your problem in a fraction of the time it would take you. 

This is usually how it happens, you figure it shouldn't take longer than a few minutes to get something up and running using AI, so you put as little effort and time as you can into prompting the AI, only to be met with broken results. Then you put a little bit more time and effort, in the hopes that with a little bit more feedback you can get this dumb machine to produce the desired result. This can, and sometimes *does*, go on for hours, and the sunk-cost fallacy doesn't present you an exit out of the endless loop.
## 2. Project idea

With my project I want to underline how relying on contemporary artificial intelligence which is incentivized to do the work for you, as opposed to coaching you to become a better, more informed version of yourself, may not only fail, but can often waste more of your time than had you endured a little bit of unproductivity at the start, learning the skills required, to later work up that lost time; often getting further, than had you allocated all your time prompting AI.

The whole project is composed of 4 elements, 1 is the overall message and wisdom that I'm trying to convey, and the other 3 are satirical flavoring.
- **Message**
	1. **AI over-reliance**: the overall lecture that treating your AI as the interface as opposed to an assistant that works along side you is both bad for your mental hygiene and time expenditure.
- **Satire**
	2. **Chatbot**: an annoying, over-controlling, late 90s inspired virtual assistant that isn't very intelligent and responds in over-flattery.
	3. **Satire posters**: various satirical posters as the end product of your work.
	4. **Privacy settings**: a tongue-in-cheek puzzle/whack-a-mole mini-game about turning off privacy-invasive features.

The mascot AI was inspired by late 90s virtual assistants such as Microsoft's Clippy and the freeware, later classified as spyware, BonziBuddy. In addition to making the experience more engaging, the alternate reality of AI being bundled with late 90s virtual mascots affords the player a certain comedic distance from reality which might help them view the situation with new eyes, as well as using Bonzi's iconically annoying voice as an added layer of annoyance to the AI (“BonziBuddy,” 2025; _Online Microsoft Sam TTS Generator_, n.d.).

The ideal playthrough would go like this:
1. The player is presented with the update notification, learning that the application they're about to use has had a recent overhaul, introducing a simpler interface in the form of a chatbot and removing the old workflow. 
2. Figures out that they're supposed to prompt the AI to transform the image snippet to fit inside the silhouette.
3. Struggle and get frustrated with communicating with the AI on what to do when it feels incredibly straight forward.
4. Before admitting defeat, explore the top menu and uncover the legacy settings.
5. Play around with the sliders and realize how complicated the system is.
6. With enough playing around, realize they can overlay the sliders in order to simplify the transformations and mold the setup in service of their use cases; teaching autonomy.
   If they don't get that far, at least realize they can combine the AI's rough capabilities with some fine adjustments using the legacy interface, learning to use it as *a* tool and not *the* tool.
## 3. Implementation

The project utilizes the optimization algorithm known as gradient descent. This algorithm, explained at a high level, uses calculus to take the derivative of a provided loss function which basically takes in input and the current state of the trained neural net and outputs a loss value, telling it how far it was from the target output. Assuming that the loss function is differentiable, the algorithm calculates the slope, which points in the direction of the steepest increase in loss (so we move in the opposite direction), and then combine the slope of each dimension into a single vector. This direction is then multiplied by a carefully picked learning rate which dictates how large the iteration steps are. These iterations are continued until you stop making progress on lowering the loss (“Gradient Descent,” 2025).

<video src="https://upload.wikimedia.org/wikipedia/commons/transcoded/4/4c/Gradient_Descent_in_2D.webm/Gradient_Descent_in_2D.webm.720p.vp9.webm" width="200" autoplay repeat/>

Using TensorFlow we don't need to implement the math as stated, instead only supply the loss function, find the optimal learning rate, define the dimensions of the model (number of hidden layers and neurons) and run the teaching algorithm until the loss value reaches a given threshold. The loss function does all the magic as it holds the task of reading the output values of the model and return a score of how far it was away from the desired output. In my case I construct a matrix transformation based on the combined output values of each output neuron and then subtract that matrix from the desired matrix and return the sum of the absolute difference as the loss (_TensorFlow_, n.d.).

ML models are based around training the model to be general enough to be able to take in a range of inputs and return a useful output. Gradient descent is only an optimization problem to help to configure the weights and biases so that the model learns to output the target value based on the given input, but in order to make it general you have to train it on a range of input data which I purposefully decided not to do. The reason for this is because I was worried the model might learn to recreate the target matrix too accurately which wasn't the experience I was going for. Therefore, making the AI visually struggle towards the target matrix unveils more of the underlying mechanisms at play.
## 4. Evaluation/Conclusion

The initial idea was supposed to be straight forward enough. I wanted to take the confusing and complex world of matrix transformations, make the transformations so convoluted that only AI would make sense of it, and then make a tool that allows the player to build intuition behind the transformations and hopefully learn something about matrix transformations as an outcome, in addition to the overall AI message.

The scope exploded fairly quickly as I felt the foundation didn't stand strong enough on its own. So driven by relentless anxieties, I added more and more flavor whenever I found the opportunity. But flavor aside, even just the AI chatbot which was gonna be driven by smokes and mirrors using simple keyword-matching ended up needing many times more keywords and commands than planned in order to provide the AI all the power needed to be useful enough on its own.

It would have been defendable to give the AI only the power of simple transformations, and towards the end require more complex and creative transformations such as skews, mirrors and stretching that could amply demonstrate AI's inability to think outside the box or diverge from the norm in general, requiring you to abandon it in order to make something truly creative; but it would require a visual piece that contains deformations of cutouts which would be really difficult both for me to produce and figure out the correlating matrix values to. So, since I only realized this once I was far down a different path, I did not change course.

In the end, while some of the satirical flavoring might be a bit extremified, heavy handed and on occasion lean more into people's collective consciousness of AI, influenced by generations of entertainment media, than the honest to truth contemporary reality; it, in the least, can add some engagement to the overall message.
## 5. Bibliography

- BonziBuddy. (2025). In _Wikipedia_. [https://en.wikipedia.org/w/index.php?title=BonziBuddy&oldid=1327347573](https://en.wikipedia.org/w/index.php?title=BonziBuddy&oldid=1327347573)
- Gradient descent. (2025). In _Wikipedia_. [https://en.wikipedia.org/w/index.php?title=Gradient_descent&oldid=1328775470](https://en.wikipedia.org/w/index.php?title=Gradient_descent&oldid=1328775470)
- _Online Microsoft Sam TTS Generator_. (n.d.). Retrieved January 12, 2026, from [https://www.tetyys.com/SAPI4/](https://www.tetyys.com/SAPI4/)
- _TensorFlow_. (n.d.). TensorFlow. Retrieved January 12, 2026, from [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Zucconi, A. (2016, February 10). The Transformation Matrix. _Alan Zucconi_. [https://www.alanzucconi.com/2016/02/10/tranfsormation-matrix/](https://www.alanzucconi.com/2016/02/10/tranfsormation-matrix/)

### Link to project: https://blog.alexanderfreyr.com/creative-use-of-ai/