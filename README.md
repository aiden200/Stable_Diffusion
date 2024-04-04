# Stable_Diffusion
Stable Diffusion from Scratch


background p_\theta(x_0:T) := p(x_T) p_\theta(x_t-1|x_t) := \Norm(x_t-1;\mu(x_t,t), 
\variance_theta(x_t,t))


Forward and backwards
in Forward, we just take the mean and variance based on beta, which represents how much "noise" we want to insert. We use a variational autoencoder to compress the image into a latent space. The latent space is in a multivariate distribution.



In the backward pass, we estimate how much noise there is in the image, and remove it. We need to estimate P_theta(x). Since the closed form solution is intractible, we maximize ELBO, the lower bound. While we are removing the noise, we uncompress the latent space.

We want the model to condition on what we want to build (like for example, we want an image of a cat).

In the unet, the model will have the current state, how much noise, and the prompt so the model can remove the noise in a way that aligns with the prompt. We can perform contrastive language-image pre-training (CLIP) the prompt so the model learns both representations. The equation is below 

Classifier Free Guidance:
output = w *(output w/ prompt - output w/out prompt) + output w/out prompt
w stands for how much you want the model to pay attention to the prompt


Our conditionals, are our embeddings of our prompt (text or image)