# Presentation Script — "Does Steepest Selection Help?"

> A slide-by-slide spoken script. Read it like you're saying it out loud.
> Anything in *(parentheses italics)* is a stage direction / aside — don't read those.
> Square-bracket numbers like **[~30s]** are rough timings.

---

## SLIDE 1 — Title

**[~20s]**

"Hi everyone, my name is Shyam Patadia. The title of my project is *Does Steepest Selection Help? An Empirical Study of Saddle-Point-Escaping Optimizers on Non-Convex Machine-Learning Loss Landscapes.*

That's a long title, so let me break it down. The whole project is about a particular optimizer called SPGD — Steepest Perturbed Gradient Descent — and the question I'm asking is a simple one: when this optimizer works, *what part of it* is doing the work? Is it the random jumping around? Or is it the rule that picks which jump to commit to? That's the question."

---

## SLIDE 2 — Roadmap

**[~30s]**

"Here's how I'll walk you through this:

First, I'll explain the actual problem — what training a model really *is* mathematically, and why something called a 'saddle point' is the main villain of the story.

Then I'll introduce SPGD itself, the algorithm we're studying, and explain what's new about it.

After that I'll point out a gap — something the original authors didn't test — and explain the experimental design I used to test it, including a key control I had to invent.

Then four experiments, walking through what each one tells us.

Finally I'll summarize where SPGD helps, where it doesn't, and how my work compares to the original paper."

---

## SLIDE 3 — Training a model is an optimisation problem

**[~60s]**

"Okay, the basics. When you 'train' a machine-learning model, what you're actually doing — mathematically — is *minimizing a function*.

The function is up on screen. It's called the **empirical risk**. Don't be intimidated by the notation — all it says is:

- You have $N$ training examples.
- For each example, you measure how wrong the model is — that's the loss $\ell$.
- You average that wrongness across all examples.
- And then you adjust the parameters $\theta$ — the weights of the model — to make that average as small as possible.

So 'training' = 'find the $\theta$ that minimizes the average loss.'

Now four important facts about this minimization problem in modern machine learning:

1. The dimension $d$ — the number of parameters — is *huge*. Millions, sometimes billions.
2. The function $f$ is **non-convex** — meaning it has lots of bumps, valleys, plateaus, not a single nice bowl.
3. We use **first-order methods** like SGD or Adam, which only know the *gradient* $\nabla f$ — the local slope.
4. And here's the key insight from research: in high dimensions, the thing that traps optimizers isn't bad local minima — it's **saddle points**. They're everywhere. They're the main obstacle.

So the next slide explains what 'stuck at a saddle' actually means, because that's the enemy here."

---

## SLIDE 4 — What does "stuck at a saddle" actually look like?

**[~90s]**

"Let me show you what a saddle point actually *is*, because the math word 'saddle' is more literal than people realise — it really does come from a horse's saddle.

So picture an actual horse's saddle. Take a ball and place it somewhere on it.

Now roll the ball *front-to-back* — along the dip where the rider would sit. It swings back and forth, oscillates, and eventually it settles right in the middle of the dip. From this direction, that middle point looks like a *minimum*. The ball is at rest. The 'gradient' — the slope of the surface in this direction — is zero. Everything tells you 'we're done.'

But here's the trick. The ball is in *unstable equilibrium*. If you give it the slightest nudge in the *perpendicular* direction — toward either side of the saddle, where the rider's legs would dangle — the surface curves *sharply down*, and the ball **falls off the saddle drastically**.

That middle point is the **saddle point**. It's a minimum in one direction and a maximum in the perpendicular direction. The gradient is zero, so a first-order method like SGD looks at it and thinks, 'I'm at a minimum, I'm done.' But there's a real descent direction available — the optimiser just can't see it because the gradient at the saddle is exactly zero in every direction.

*(point to the figure on the right)* The figure on the right shows this in 1-D: the top panel is a cross-section through the saddle's stable direction — looks flat near the middle. The bottom panel shows the consequence: the gradient norm drops below this threshold $\varepsilon$ and stays there. That's our 'we're stuck' detector — a stagnation episode.

In high dimensions — millions of parameters — this gets way worse. Every saddle has *some* directions where it looks like a minimum and *some* where it's actually a maximum. There are exponentially more saddles than true minima. That's why saddles dominate the landscape and that's why they trap us."

---

## SLIDE 5 — Two ways to get unstuck

**[~100s]**

"So if gradient methods get stuck at saddles, how do we get unstuck? There are basically two families of answers — and let me explain them with something everyone in this room uses every day: a recommendation system.

*(point to the recommender block)* Imagine you're Spotify. A user is listening, and you've been recommending songs you think they'll like. But they just keep hitting skip. Your model is *stuck* — its best guess is no longer making them happy, and the obvious next-best song is too similar to the one they just skipped. What do you do? There are two strategies.

**Strategy 1: Tune what you already know.** This is the **adaptive** family — Adam, RMSProp, AdaGrad. The idea is: look at the user's history *per feature*. Maybe they always finish acoustic songs — so be careful and take small, fine-grained steps in that direction. Maybe they always skip electronic songs — so take *big* steps to move away from that region quickly. Each feature gets its own scaling, based on how confident you are.

That's literally what Adam does for gradient descent. It rescales the gradient per parameter, individually, based on the history of past gradients. Steep directions get smaller steps; flat directions get bigger steps. That's why Adam usually wins on benchmarks — it's just *smarter* about how to step.

But — here's the catch. Adam still only uses information it has already collected from the user. It never says, 'let me try something completely different and see what happens.'

**Strategy 2: Play a wildcard.** This is the **perturbation** family. When the model is stuck, the strategy is: don't trust the history. Just *recommend a totally random song* from the catalogue — even one that doesn't fit the user's pattern at all.

Most random recommendations will be skipped. The user has never liked this stuff before, why would they now? But occasionally — by accident — that random song will be a banger they never would have discovered from their existing taste. And then suddenly your model has a whole new region of preferences to learn from.

PGD — Perturbed Gradient Descent — does exactly this for optimisation. When the gradient gets small and you suspect you're stuck, throw in some Gaussian noise. Most of the noise lands you nowhere useful. But occasionally it lands you somewhere with a meaningful gradient again, and you can keep descending. The catch with PGD is: it commits to *one* random pick. Whatever that one pick was, you live with."

---

## SLIDE 6 — Why does "take a leap" actually work?

**[~75s]**

"Why should a random jump help at all? Let me give you the geometric reason, because this is actually beautiful.

A 'strict' saddle has what mathematicians call a *negative curvature direction* — I'm calling it the **tail**. It's a direction where the loss actually *decreases*, but the slope is exactly zero right at the saddle, so the gradient sees nothing.

Here's the magic: **a random direction will, with positive probability, have some component along that tail.** Once you're nudged onto the tail even a tiny bit, the gradient becomes informative again, and you slide down.

Concrete example in 1-D: imagine a flat plateau next to a downhill slope. Jump left or right at random — you've got a 50/50 chance of landing on the slope.

In high dimensions it's actually *better*, not worse. The tail directions occupy a measure-positive fraction of the unit sphere — meaning random isotropic noise will hit a useful direction with overwhelming probability. So perturbation provably works at escaping saddles. PGD has a theorem for this — it escapes in polynomial time.

*(pause for the [pause])*

Okay so perturbation works. But here's the question that nobody asked before SPGD:

**Among all the random jumps you could take, which one should you actually commit to?** PGD just takes whichever one it sampled first. But what if you sampled *several* and picked the best?

That's the gap SPGD fills."

---

## SLIDE 7 — SPGD, intuitively

**[~60s]**

"This is SPGD in one line.

PGD takes one random jump and hopes for the best. **SPGD takes $N_P$ random jumps, evaluates all of them, and commits to the one that lowers the loss the most.**

That's the whole idea. 'Look before you leap.'

*(point to left block)* Why this should help: a single random jump might land you back on the same plateau, or — even worse — *up* the saddle's slope, making your loss go up. But if you sample, say, 5 candidates, you're way more likely to find one that lands you in a genuinely lower spot. And SPGD only commits *if* the best candidate beats your current position. So in the worst case, nothing happens — you don't get hurt.

*(point to right block)* Where SPGD sits philosophically: think of it as a middle ground between two extremes. Plain gradient descent is super directed — it always knows which way to go — but never explores. Random search is pure exploration — it tries everything — but has no direction. SPGD blends them: gradient gives you *direction*, the candidate sampling gives you *exploration*.

And once we're done with intuition, the actual algorithm is just this idea written down precisely. Let's look at it."

---

## SLIDE 8 — Three differences from PGD

**[~50s]**

"Before the pseudocode, here are the three concrete differences between SPGD and PGD:

**1. When?** PGD waits — it only perturbs *when* it detects you're stuck (gradient is small). SPGD doesn't wait. Every $\textit{IterP}$ steps — say, every 200 steps — it perturbs *unconditionally*, whether you're stuck or not. It's proactive.

**2. How many?** PGD draws *one* perturbation. SPGD draws $N_P$ candidates — say, 5 or 8 of them.

**3. Which one?** PGD just uses the one it drew. SPGD evaluates all $N_P$ and commits the one with the steepest loss decrease — and only if that one is actually better than your current position. That's the paper's '$\le$ acceptance rule.'

That third difference — *the selection rule* — is the one we'll spend the whole project asking about. Is it doing useful work, or could you get the same benefit by just picking a candidate at random?"

---

## SLIDE 9 — The SPGD algorithm

**[~3 min]**

"Okay, now I want to walk through the actual algorithm slowly, line by line, with a concrete real example — so you really understand what each part does, and more importantly, **which lines are SPGD-unique versus which lines are just plain gradient descent.** Most of this algorithm is *not* new; only a small block in the middle is.

### Let me set up the example.

Imagine we're training **ResNet-18 on CIFAR-10**. CIFAR-10 is a classic dataset — 50,000 small images of 10 categories: cats, dogs, planes, cars, etc. ResNet-18 is a deep convolutional network with about **11 million parameters**. That whole 11-million-element vector is our $\theta$. This is literally Experiment 4 in my paper, so the numbers I'm about to use are real.

### The Require line — what you give SPGD before it starts.

*(point to top of algorithm)* Look at the top — these are the inputs:
- $\theta_0$ — the **initial weights** of ResNet-18, randomly initialized.
- $\eta$ — the **learning rate**, we use 0.01. Standard SGD parameter.
- $Amp$ — the **perturbation amplitude**, 0.001. This means when we perturb a weight, we shift it by at most 0.001 in either direction.
- $N_P$ — **number of candidates** per perturbation phase. We use 5.
- $IterP$ — **how often perturbation fires**. We use 200, meaning every 200 minibatch updates — roughly twice per epoch on CIFAR-10.
- $T$ — total training steps.

The first three — $\theta_0$, $\eta$, and the loop length — are exactly what *plain SGD* needs. The last three — $Amp$, $N_P$, $IterP$ — are the **SPGD-specific knobs.** Three new hyperparameters, that's the whole API surface.

### The loop — let me walk through what happens at different steps.

For each step $t$ from 0 up to $T-1$, *two things might happen.*

**On most steps — say step 1, step 2, step 50, step 199 — the only thing that runs is line 11.** Look at line 11. That's $\theta_{t+1} \leftarrow \theta_t - \eta \nabla f(\theta_t)$. Take the gradient of the loss with respect to the weights, and subtract a small fraction of it. **This is exactly vanilla SGD.** Nothing new here. If you only ever ran line 11, you'd have plain stochastic gradient descent — the optimizer everyone has been using since 1951.

**But every 200 steps — at step 200, 400, 600, and so on — the `if` on line 2 fires.** Now we enter the perturbation phase, which is **lines 3 through 8**, *before* the standard gradient step on line 11. **Lines 3 through 8 are the SPGD-unique part.** This block is the entire contribution of the paper. Let me walk through what happens at, say, step 200 of our CIFAR-10 training:

#### Line 4 — Sample $N_P$ candidates.

We generate 5 candidate weight vectors. Each candidate is 'the current ResNet-18 weights, plus a random offset.' For every one of the 11 million parameters in the network, we draw a uniform random number in $[-0.001, +0.001]$ and add it. We do this five times, independently. **So now we've got 5 slightly perturbed copies of the network sitting in memory** — five alternative ResNet-18s, all very close to where we currently are, but each pushed in a different random direction.

#### Line 5 — Evaluate each candidate.

For each of those 5 candidates, we run **one forward pass** on the current minibatch and record the loss. So that's 5 extra forward passes total. **Important: no extra backward passes. No extra gradients are computed here.** That matters for cost — gradients are roughly twice as expensive as forward passes, and SPGD avoids that overhead. We only need *function values*, not derivatives, to rank the candidates.

#### Line 6 — Pick the best.

Look at the 5 losses, pick the candidate with the smallest one. That's $k^\star$. The little star just means 'the winner.'

#### Line 7 — The acceptance check.

This is the most important line in the whole algorithm. We ask: '**Is the winning candidate's loss actually $\le$ the loss at our current weights?**' If yes, jump there — line 8 — replace the current weights with that perturbed version. If no — meaning even our best candidate is worse than where we already are — **do nothing.** Stay put.

This is what's called a **no-regret property**: SPGD can never make things actively worse. The worst-case outcome of a perturbation phase is that you wasted 5 forward passes and stayed put. You don't accidentally jump to a bad spot.

#### Line 11 — Gradient step (back to vanilla SGD).

Whether or not we accepted a candidate, we then do the standard SGD step from wherever we are now. Gradient flows from the current position — possibly the perturbed position, possibly the original — and we keep training.

### Then steps 201 through 399 are pure SGD again.

The `if` on line 2 only fires every $IterP = 200$ steps, so for the next 199 steps, we just run line 11 over and over. Plain gradient descent. Until step 400 — when the cycle repeats.

### So zooming all the way out:

**SPGD is identical to SGD on 199 out of every 200 steps.** On the 200th step, it inserts a small Monte-Carlo exploration block *before* the gradient update — sample 5 perturbations, evaluate them, pick the best, jump only if it's better. That's the entire algorithm. That's everything that's new.

### Cost — the two bullet points on the slide.

*(point to bullets)*

**First bullet — cost per perturbation phase.** Each phase costs $N_P$ extra forward passes. In our case, 5. No extra gradients. So if a forward pass costs $F$ and a backward pass costs about $2F$, then a normal training step costs $3F$, and a perturbation step costs an extra $5F$ on top of that. Sounds expensive — but it only happens once every 200 steps, so the average overhead is **under 1%.** Effectively free.

**Second bullet — the two knobs that matter.** $N_P$ controls how *broadly* you explore each phase — more candidates means a better chance of finding a good one, but more compute. $IterP$ controls how *often* you explore — smaller $IterP$ means more frequent perturbation. We do a full $4 \times 4$ ablation on both of these in Appendix A — I'll mention it in the backup slides if there are questions.

So that's SPGD, completely unpacked. **The vast majority of the algorithm is just SGD you already know.** The novelty is a small targeted insertion: every $IterP$ steps, take $N_P$ random looks at nearby positions and commit to the best one *only if* it actually improves things."

---

## SLIDE 10 — What did the original paper test?

**[~70s]**

"Now I want to set up the gap in the literature, because that motivates everything I did.

The original SPGD paper, by Vahedi and Ilies in 2024, did test their algorithm — but on a fairly narrow set of problems.

*(point left)* They tested on 4 smooth 2-dimensional benchmark functions — Peaks, Ackley, Easom, Levy — these are the kind of toy functions optimization researchers use because you can plot them. Plus a 3-dimensional engineering problem, packing components into a box.

*(point right)* And they compared SPGD against plain Gradient Descent, PGD, Simulated Annealing, and MATLAB's `fmincon`.

What's *missing*?

**First**, no neural-network experiment. None at all. This matters because neural-network loss surfaces are wildly different from smooth 2-D test functions — they have ReLU kinks, batch normalization effects, mini-batch noise, all kinds of weird geometry.

**Second**, and this is methodologically the bigger problem — there's no random-selection control. They show 'SPGD beats PGD,' but PGD is a totally different algorithm. They never compared SPGD against *the same algorithm with the selection rule turned off.* So you can't tell whether the *selection* is doing the work, or whether just *having more candidates* is doing the work.

*(pause for [pause])*

And yet — the paper's own conclusion section says SPGD has 'potential to significantly enhance neural network training.' That's the claim we're going to test."

---

## SLIDE 11 — Two questions only experiments can answer

**[~45s]**

"So I distilled the gap into two questions.

**Question 1: Does the paper's neural-network claim hold?** They said it 'has potential.' Does it? When? By how much? You can't answer that with theory. You have to run it.

**Question 2: Is it the selection rule, or just the perturbation?** SPGD has two ingredients — multi-candidate perturbation, and steepest selection. The original paper bundles them. To separate them, you need a control that perturbs the same way but selects *randomly* instead of by best loss.

That control is the central methodological contribution of this work. I built it and called it RPGD — Random Perturbed Gradient Descent. Let me show you how it fits in."

---

## SLIDE 12 — The five optimizers we compare

**[~75s]**

"So here are all five optimizers we put head-to-head. Let me explain each row.

**SGD** — Stochastic Gradient Descent. Plain vanilla. The bare-minimum baseline. No tricks.

**Adam** — The adaptive method I described earlier. The default in modern ML. We include it to ground our results against current practice.

**PGD** — Standard perturbed gradient descent. One Gaussian perturbation when the gradient gets small.

*(emphasize)* **RPGD** — This is the key one I introduced. It uses the *same* multi-candidate perturbation mechanism as SPGD — same number of candidates, same uniform sampling, same compute cost. But instead of picking the steepest, it picks one *at random*.

**SPGD** — The full algorithm. Steepest of $N_P$ candidates, paper's acceptance rule.

*(point to the contrasts block)* And here's why each pairwise comparison matters:

- **SPGD vs. RPGD** — same compute, same mechanism, only the selection differs. So if SPGD wins, it's the *selection rule* doing the work. This is the novel comparison.
- **SPGD vs. PGD** — does multi-candidate perturbation help at all over a single perturbation?
- **SPGD vs. SGD** — does perturbation+selection beat doing nothing fancy at all?"

---

## SLIDE 13 — Paired-seed protocol + saddle diagnostics

**[~75s]**

"Two methodological details that matter a lot for clean results.

*(point left)* **Paired-seed protocol.** Random seeds matter a *huge* amount in deep learning — different initializations can give wildly different final accuracies. So the standard approach is: run each optimizer many times with different seeds and average. But then you're comparing averages across *different* seeds, and the seed noise can swamp the optimizer effect.

What I do instead: for each seed, fix the data split, fix the initialization — and *then* run all five optimizers from that *exact same starting point*. So at every seed I get five outcomes that share initialization. I can subtract them — that's a 'paired comparison.' It cancels out the seed noise. With small seed counts like 3 or 5, this is the difference between seeing a real effect and seeing nothing.

*(point right)* **Direct saddle diagnostics.** I don't want to *infer* whether saddles matter — I want to *measure* it. So I track:

- A **stagnation episode** — a stretch of consecutive steps where the gradient norm is below some tiny threshold $\varepsilon$. If you see lots of stagnation, saddles are biting.
- **Escape time** — first step where the loss drops 5% below its starting value. This tells you how long you sat in the saddle plateau before getting out.

So instead of saying 'maybe saddles caused this,' I can say 'the gradient norm was below threshold for X steps.' Concrete. Measurable."

---

## SLIDE 14 — Six experiments

**[~45s]**

"Six experiments total. Four are 'discriminating' — meaning they're designed to actually tell optimizers apart. Two are diagnostic — they're sanity checks that ended up not discriminating, but they reveal interesting secondary stuff, so I put them in the appendix.

The four main ones are:

- **Experiment 1** — Synthetic non-convex test functions, 30 seeds each, dimensions 2, 10, and 50. Classic optimizer benchmarks.
- **Experiment 3** — Real tabular data from OpenML, 5 datasets, MLP classifier.
- **Experiment 4** — The big one. CIFAR-10 image classification with ResNet-18. This is the anchor experiment.
- **Experiment 6** — MovieLens matrix completion. This is the one with documented saddle-point structure, where SPGD's claim should *really* shine if it ever does.

The scale is increasing dramatically — from 2-dimensional toy problems up to 11 million parameters in ResNet-18."

---

## SLIDE 15 — Experiment 1: Synthetic benchmarks

**[~75s]**

"Experiment 1. The classics.

*(point left)* I ran all five optimizers on three benchmark functions — Rastrigin, Ackley, Rosenbrock — at three dimensions: 2, 10, and 50. Thirty random initializations per cell. Convergence is declared when the loss drops below $10^{-2}$.

These functions are evil on purpose. Rastrigin and Ackley both have *tons* of local minima — like an egg carton. Vanilla gradient descent gets trapped immediately. Rosenbrock is a long curved valley.

**Findings:**

- The cleanest result is on Ackley at $d=10$. SPGD's mean final loss is $-2.57$ below RPGD's, on a paired comparison. That's a real signal — same starting points, same compute, just the selection rule differing, and SPGD wins.
- At $d=50$ the perturbation methods start *failing*. Why? The amplitude $\textit{Amp}$ was constant — but in higher dimensions, a uniform perturbation of fixed amplitude per coordinate produces a much bigger total displacement (it scales like $\sqrt{d}$). So a perturbation that was helpful at $d=2$ becomes destructively large at $d=50$.
- The fix is per-parameter scaling of the amplitude — using each parameter's standard deviation. I applied that in all later experiments.

*(point right)* The success-rate plot on the right shows: SPGD matches or exceeds RPGD on every Rastrigin and Ackley cell."

---

## SLIDE 16 — Experiment 3: OpenML-CC18 tabular

**[~60s]**

"Experiment 3 moves us to real data — five tabular classification datasets from the OpenML-CC18 benchmark suite. Different sizes, different feature types — credit scoring, software-defect prediction, spam detection, image segments, vehicle silhouettes.

A 3-layer MLP, three seeds per dataset per optimizer.

**Findings:**

- SPGD's mean test accuracy is above RPGD's on a *majority* of paired (dataset, seed) comparisons. So the synthetic-benchmark signal carries over to real heterogeneous data.
- But — and this is honest — Adam dominates absolute accuracy on all five datasets. The reason is that tabular data with a small MLP is a *well-conditioned* problem. The loss surface doesn't have many saddles. There's no plateau to escape from. Adam's adaptive scaling just chews through it faster than anything that involves perturbation.
- We almost never saw stagnation episodes here. Only two datasets, credit-g and vehicle, had any at all, and just a handful per run. Confirms: this regime isn't where SPGD's mechanism matters.

So the result is: SPGD slightly beats its random control, confirming the rule helps, but the absolute winner here is Adam because the landscape is too easy to need perturbation."

---

## SLIDE 17 — Experiment 4: CIFAR-10 / ResNet-18 (the anchor)

**[~120s]**

"This is the headline experiment. Pay attention.

*(point top left)* **Setup.** Real deep learning. ResNet-18 — 11 million parameters — trained on CIFAR-10 image classification. Fifty epochs, three seeds. I ran this on the WPI Turing cluster on A30 GPUs.

One important design choice: I deliberately disabled batch normalization in the early layers — `layer1` and `layer2`. Why? Because batch norm has a smoothing effect on the loss landscape. It tends to *erase* saddle points, which would make this experiment uninformative for SPGD. Disabling BN in the early layers preserves the saddle structure I want to test against.

*(point at the table)* **Results — final test accuracies:**

- Adam: $0.90$. The clear absolute winner.
- SPGD: $0.83$.
- SGD: $0.83$ — basically tied with SPGD overall.
- PGD: $0.81$.
- RPGD: $0.79$. The worst.

That ordering already hints at what's happening. But raw averages don't tell the full story — three seeds is small, the unpaired standard deviations are wide. Look at the *paired* plot on the right.

*(point at the figure)* **The headline.** When I subtract per-seed: SPGD beats RPGD on *every single seed* — by 4.4, 1.1, and 5.8 percentage points. The paired mean is **+3.78 percentage points test accuracy**.

That is the cleanest evidence in the entire study that **the steepest-selection rule is doing useful work.** Both algorithms perturb the same way, both pay the same compute cost, both share an initialization on every seed. The only difference is one picks the best candidate, the other picks a random one. And the best-picker wins by 3.78 points consistently.

This is the result. If you remember nothing else from this presentation, remember this slide."

---

## SLIDE 18 — Experiment 6: MovieLens matrix completion

**[~120s]**

"Experiment 6 is where I directly stress-test the original paper's claim about neural-network training.

*(point top left)* **Setup.** I take the MovieLens-100K rating matrix — users by movies, $943 \times 1682$, with most entries missing. I try to fill in the missing ratings via a low-rank factorization: $U V^\top$ where $U$ and $V$ are skinny matrices of rank 5. This is called Burer–Monteiro factorization.

Why this benchmark? Because the optimization theory community has *proven* that this problem has a saddle point near the origin. If you initialize $U$ and $V$ small, you start *inside* the saddle's region, and gradient descent escapes only slowly. This is exactly the geometry SPGD's selection rule is designed for. So if SPGD ever shines, it should shine here.

*(point right)* And it does — sort of. Look at the curves. The training loss on the left shows SPGD pulling ahead of SGD through the saddle plateau. SPGD escapes 50 steps earlier than SGD on every seed.

*(point to the two key numbers)* **Two important numbers:**

**1. Acceptance rate.** SPGD accepts a candidate on **30%** of perturbation phases. RPGD only accepts on **9%**. That threefold gap is *direct evidence* that the selection mechanism is firing as designed. It's not a silent algorithm. The selection rule is genuinely picking better candidates.

**2. Escape time.** SPGD escapes the saddle 50 steps before SGD, 40 steps before RPGD, on every seed.

*(point to bottom block)* **But — and this is the honest part — the result is split.**

The mechanism *transfers*. In a 13,000-parameter non-convex ML problem, SPGD's selection rule is empirically firing as the paper specifies. So the paper isn't wrong about the algorithm working in this regime.

But the optimization head-start does *not* translate into a held-out generalization gain. Test RMSE is statistically tied with SGD. So 'SPGD makes you generalize better' — that part doesn't show up here.

I call this **mechanism-positive, generalization-neutral**. The selection rule does what the paper says, but the downstream benefit the paper hypothesizes — better test performance — doesn't materialize on this benchmark."

---

## SLIDE 19 — Cross-experiment summary

**[~60s]**

"Let me bring it all together in one table.

Across all four discriminating experiments:

- **Synthetic Ackley at $d=10$**: SPGD's selection rule helps. Paired loss delta of $-2.57$.
- **OpenML tabular**: Slight edge for SPGD over RPGD, around 1 percentage point.
- **CIFAR-10 / ResNet-18**: This is the strong one. Paired $+3.78$ percentage points test accuracy, SPGD beats RPGD on 3 out of 3 seeds.
- **MovieLens matrix completion**: The selection rule fires three times as often as the random control. SPGD escapes the saddle plateau 50 steps earlier on every seed.

The pattern is consistent: the steepest-selection rule provides a measurable advantage over same-mechanism random perturbation across all four settings.

*(pause for [pause])*

The advantage is **mechanism-positive on every setting**, and **generalization-positive on CIFAR-10 specifically**."

---

## SLIDE 20 — Where the algorithm doesn't help

**[~75s]**

"I want to be honest about three places SPGD's story breaks down. Otherwise this would just be a sales pitch.

**One: the saddle-escape framing is over-stated for modern ML.** Across all 15 CIFAR-10 runs — five optimizers times three seeds — *no* optimizer ever had a stagnation episode. Not one. The gradient norm never sat below $10^{-3}$ for five consecutive steps. And remember, I deliberately handicapped the network by disabling batch norm in the early layers to *preserve* saddle structure. Even with that handicap, modern ML training simply doesn't sit at saddles for measurable periods. So whatever advantage SPGD got on CIFAR-10, it can't be from 'escaping plateaus' literally — it's something subtler.

**Two: generalization gain on matrix completion is null.** SPGD's training-time lead doesn't survive to held-out RMSE. So in this regime, SPGD is a faster optimizer but not a better generalizer.

**Three: Adam dominates absolute accuracy throughout.** Adam's per-parameter adaptive scaling implicitly preconditions the loss landscape — it's doing geometric work that none of our perturbation methods do. So Adam wins. But that's not a fair comparison to SPGD — SPGD's contribution is *the selection rule*, and the right baseline for that is RPGD, not Adam."

---

## SLIDE 21 — Comparison with the original paper

**[~60s]**

"Here's a side-by-side comparison of what the original paper did versus what I did.

- **Domain.** They tested on 2-D test functions and a 3-D engineering problem. I tested on neural networks and matrix completion.
- **Optimizers compared.** They compared against GD, PGD, Simulated Annealing, and MATLAB's `fmincon`. I compared against SGD, Adam, PGD, and crucially RPGD.
- **Random control.** Absent in their paper. Introduced here.
- **Saddle diagnostic.** They inferred saddle behavior from loss curves. I measured it directly with stagnation episodes and escape time.
- **Paired protocol.** They didn't use one. I used it across every experiment.
- **Compute scale.** They ran in seconds on 2-D problems. I ran ResNet-18 on a GPU cluster.

The bottom line: they *proposed* the algorithm. I provided the first **controlled, paired, ML-scale test** of its central claim — and the only same-compute random-selection ablation that exists in the literature."

---

## SLIDE 22 — Three things this study contributes

**[~75s]**

"Three contributions, in priority order.

**One: the first ML-scale test of SPGD.** The original paper actually *names* neural-network training as an in-scope use case in its conclusion, but the paper itself contains no NN experiment. I cover that gap with four experiments spanning synthetic, tabular, vision, and matrix completion — up to 11 million parameters.

**Two: the RPGD control.** This is the methodological contribution. Without a same-compute random-selection baseline, you literally cannot tell whether 'SPGD beats baselines' means 'the selection rule works' or 'just having multiple candidates works.' On CIFAR-10 the SPGD-vs-RPGD paired delta is $+3.78$ points on every seed — a result that *could not have been measured* by the original protocol. RPGD makes that measurement possible.

**Three: bounded scope claims, supported by direct diagnostics.** I'm not claiming SPGD is the new best optimizer. I'm saying: it helps where the selection rule fires, like CIFAR-10 and Ackley $d=10$, and it's neutral where the optimization gain doesn't translate, like MovieLens RMSE. Stagnation episodes and escape time give me the receipts for these claims — they're measurements, not assertions."

---

## SLIDE 23 — What we conclude

**[~60s]**

"My main empirical claim: on modern ML loss landscapes, **SPGD's value is best framed as a biased candidate-selection rule, not as a saddle-escape mechanism.** The framing the original paper uses — about escaping saddles — is over-stated. We didn't see saddles biting in our deepest experiment. But the *selection rule itself* still helps, even when there's no obvious saddle to escape from. So that's the right way to think about SPGD going forward.

The methodological recommendation that comes out of this: **future perturbation-based optimizers should always be benchmarked against a same-compute random-selection control.** That comparison is missing from much of the literature. Without it, you can't separate 'the noise helped' from 'the smart selection helped.'

And the one-line take-away from all this work, the line I'd put on a sticker:

> **It's not the noise. It's the choice.**"

---

## SLIDE 24 — Limitations

**[~60s]**

"Brief limitations to flag before I take questions:

- **Three seeds at the CIFAR-10 scale is small.** The paired protocol mitigates this, but the unpaired standard deviations are wide.
- **Stagnation threshold is fixed.** A relative threshold — say, 'gradient norm dropped 100x' — might catch plateaus my absolute threshold misses.
- **No momentum or learning-rate schedule on SPGD.** I deliberately left those out to isolate the selection rule's effect. Adding them would change SPGD's profile in ways worth studying.
- **Constant amplitude.** A per-parameter amplitude proportional to each parameter's initialization scale would likely improve $d=50$ and ResNet-18 results.
- **No flat-minima or sharpness measurement.** I scoped that out because the zero-stagnation result made it less central, but it's a natural follow-up."

---

## SLIDE 25 — Thank you

**[~10s]**

"Thank you. I'm happy to take questions."

*(stop here. the backup slides are for Q&A — only flip to them if asked.)*

---

## BACKUP SLIDES — only flip to these if asked

### Backup 1 — Why $\varepsilon = 10^{-3}$ on noisy gradients?

"On smooth full-batch losses like the synthetic benchmarks, the gradient decays smoothly to around $10^{-4}$ near minima, so I used the proposal's $\varepsilon = 10^{-4}$. But on mini-batched problems like CIFAR-10 and OpenML, the gradient norm is dominated by mini-batch noise — typical norms are $10^{-2}$ — so a $10^{-4}$ threshold would *never* fire and we'd see no stagnation by definition. I picked $10^{-3}$ as a consistent threshold well below the typical mini-batch noise floor. The qualitative CIFAR-10 result — zero stagnation — is threshold-dependent, and I flag that as a limitation."

### Backup 2 — Hyperparameter ablation

"I swept $N_P$ from 1 to 20 and $\textit{IterP}$ from 5 to 100 — 16 cells, 5 seeds each — on Two Moons. Compute scales as $N_P / \textit{IterP}$, so the range is about 400-fold. The accuracy heatmap is mostly flat: SPGD is robust to hyperparameter choice in this range. The loss heatmap shows that smaller $\textit{IterP}$ — more frequent perturbation — reaches lower training loss without an accuracy gain, which suggests SPGD is finding *broader* basins, not deeper ones."

### Backup 3 — Two Moons trajectories

"This was a visualization-only diagnostic. The dataset is too benign for any of the optimizers to differentiate on accuracy — they all hit ~97%. But projecting the weight trajectories into a shared 2-D PCA basis reveals something interesting: Adam moves substantially along PC1, while SGD, PGD, RPGD, and SPGD all move predominantly along PC2. That's a direct visualization of how adaptive scaling explores directions the gradient methods don't. SPGD's path is faintly serrated — those small lateral excursions are its perturbation phases."

---

## ANTICIPATED Q&A

**Q: Why didn't you compare SPGD with momentum?**
"I deliberately held out adaptivity to isolate the selection rule. SPGD plus momentum is interesting but conflates two improvements. That's a natural follow-up."

**Q: Isn't 3 seeds too few for CIFAR-10?**
"Yes, the unpaired stds are wide. The paired protocol is what saves it — paired across seeds, the SPGD-vs-RPGD gap is consistent on every single seed. That's stronger evidence than a small unpaired mean."

**Q: Why is Adam so much better and what does that say about SPGD?**
"Adam is doing structurally different work — per-parameter geometric preconditioning. It's not a fair head-to-head with SPGD's selection rule. The right comparison is SPGD vs. RPGD at equal compute. Combining SPGD's selection rule with Adam-style adaptivity is a natural next step but wasn't the question I was asking."

**Q: You said zero stagnation. Doesn't that mean saddles aren't really the issue?**
"Yes, and that's exactly the point of the 'mechanism-positive but framing-overstated' conclusion. The saddle-escape *framing* is over-stated for modern ML training. But the selection rule itself still helps even without obvious saddles. So I'm reframing what SPGD is *for* — it's a biased selector, not specifically a saddle-escaper."

**Q: How long did the experiments take?**
"CIFAR-10 was about 7.6 minutes per run on an A30, times 15 runs as a Slurm array. The synthetic and tabular experiments are seconds-to-minutes on a laptop. MovieLens is about 4 seconds per run."

**Q: What's the most surprising finding?**
"The acceptance-rate gap on MovieLens — 30% vs 9%. That's the cleanest direct evidence that SPGD's mechanism is empirically active even on a high-dimensional ML problem. It's not just a theoretical algorithm — it's actually firing."
