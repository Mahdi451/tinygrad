# Beginner Handbook

This guide is for someone who is comfortable with Python but new to machine learning and new to tinygrad.

The goal is simple:

1. Understand what this repo is.
2. Run a small example without drowning in jargon.
3. Build a mental model of how tinygrad works.
4. Make safe first contributions later.

You do not need to understand every file at once. Tinygrad is much easier to learn when you treat it like a stack of small ideas.

## Tinygrad In Very Simple Terms

Imagine you have a box of numbers.

In tinygrad, that box is called a `Tensor`.

You can:

- add boxes of numbers together,
- multiply them,
- reshape them,
- feed them through a model,
- and ask tinygrad to learn better numbers over time.

Machine learning words in plain English:

- **Tensor**: a box of numbers.
- **Model**: a recipe made of math.
- **Weights**: the numbers inside the recipe that can learn.
- **Training**: trying the recipe, seeing how wrong it was, then nudging the weights.
- **Inference**: using the trained recipe to make a guess.
- **Loss**: a number that says how wrong the guess was.
- **Gradient**: a hint about which direction each weight should move.
- **Optimizer**: the thing that actually updates the weights.
- **Backend**: the engine tinygrad uses to run the math, like CPU, Metal, CUDA, AMD, or a pure Python emulator.

Tinygrad is not a normal web app. It does not center around routes, databases, or services. It is a small deep-learning engine and library.

## What Lives Where

These are the top-level places that matter most:

- `tinygrad/`: the core library.
- `examples/`: runnable scripts that show how to use the library.
- `test/`: the behavior spec for the project. If you want to know what something is supposed to do, tests are often the clearest answer.
- `docs/`: user and developer docs.
- `extra/`: experiments, hardware tools, profiling helpers, and advanced side paths.
- `.github/`: CI workflows, useful for seeing how the project is expected to run in clean environments.

You do not need `extra/` to understand the main library.

## The Five Files To Read First

If you only read five things, read these in this order:

### 1. `docs/quickstart.md`

This gives you the basic vocabulary:

- what a tensor is,
- how tinygrad code looks,
- what training means,
- and what evaluation means.

It is the best first read if you are weak on ML words.

### 2. `examples/beautiful_mnist.py`

This is the cleanest end-to-end example in the repo.

It shows:

- loading data,
- building a small model,
- training it,
- measuring accuracy.

If you understand this file, you understand the happy-path tinygrad workflow.

### 3. `tinygrad/tensor.py`

This is the front door.

Most of the project feels complicated until you realize that almost everything starts with `Tensor`.

This file answers:

- what a tensor can do,
- when work is lazy,
- when work is actually executed,
- how gradients attach to computations.

### 4. `tinygrad/engine/schedule.py`

This file answers:

- how tinygrad takes one big math graph,
- and breaks it into smaller jobs that can really run.

This is where "nice tensor code" starts becoming "machine work".

### 5. `tinygrad/engine/realize.py`

This is the runner.

It takes scheduled work, lowers it into runnable programs or copies, and executes it.

This file answers:

- what finally makes the lazy math happen,
- how device programs are launched,
- and how performance stats are tracked.

## Your First Safe Learning Path

Use this order. Do not skip ahead too fast.

### Step 1: Install tinygrad from source

```bash
python3 -m pip install -e .
```

This makes the repo importable as a Python package while still pointing at your local checkout.

### Step 2: Run the smallest useful smoke test

```bash
PYTHONPATH=. PYTHON=1 python3 test/test_tiny.py TestTiny.test_plus
```

Why this command first:

- it is small,
- it exercises the library,
- it does not require a real GPU,
- it uses the pure Python backend, which is easier to reason about.

If this works, your basic setup is alive.

### Step 3: Run the simplest null-backend test group

```bash
python3 -m pytest test/null/
```

The `null` tests are helpful because they focus on library logic more than real hardware execution.

### Step 4: Run the main beginner example

```bash
PYTHONPATH=. python3 examples/beautiful_mnist.py
```

This is your first real "train a model" path.

### Step 5: Try the docs example

Read `docs/quickstart.md`, then compare it to `examples/beautiful_mnist.py`.

The quickstart teaches the ideas. `beautiful_mnist.py` shows the same ideas in a complete script.

## The Main Story: `examples/beautiful_mnist.py`

This file teaches a small neural network to read handwritten digits.

Open `examples/beautiful_mnist.py` and read it in chunks.

### Imports

The imports tell you the whole story:

- `Tensor`: the main data structure.
- `TinyJit`: speed helper for repeated functions.
- `nn`: layers and optimizers.
- `GlobalCounters`: performance counters.
- `function`: wraps a computation as a reusable tinygrad function.
- `getenv`, `colored`, `trange`: configuration, pretty output, progress bar.
- `mnist`: dataset loader.

### The `Model` class

The model is just a normal Python class. Tinygrad does not need a special base class like `nn.Module`.

That is important: tinygrad tries to keep the framework small and Pythonic.

### The layers list

The model is a sequence of image-processing steps:

1. Convolution
2. ReLU
3. Convolution
4. ReLU
5. BatchNorm
6. MaxPool
7. Convolution
8. ReLU
9. Convolution
10. ReLU
11. BatchNorm
12. MaxPool
13. Flatten
14. Linear

You can read this as:

"look for patterns in the image, squash values below zero, shrink the image a bit, then finally produce 10 scores, one for each digit."

### The shape story

This is the key beginner idea.

The input image starts like this:

- `B x 1 x 28 x 28`

where:

- `B` = batch size,
- `1` = one grayscale channel,
- `28 x 28` = image height and width.

Then the model changes the shape:

- after `Conv2d(1, 32, 5)`: `B x 32 x 24 x 24`
- after second `Conv2d(32, 32, 5)`: `B x 32 x 20 x 20`
- after first `max_pool2d`: `B x 32 x 10 x 10`
- after `Conv2d(32, 64, 3)`: `B x 64 x 8 x 8`
- after second `Conv2d(64, 64, 3)`: `B x 64 x 6 x 6`
- after second pool: `B x 64 x 3 x 3`
- after flatten: `B x 576`
- after final linear layer: `B x 10`

That final `10` means "score for digit 0 through 9".

### `__call__`

`__call__` says how the model processes an input.

`x.sequential(self.layers)` just means:

"start with `x`, then run each layer one after another."

### `train_step`

This is one training round.

It does:

1. `opt.zero_grad()`
   Throw away old gradient information from the previous step.

2. Pick random images with `Tensor.randint(...)`
   This creates a random batch.

3. `self(X_train[samples])`
   Run the model on those images.

4. `.sparse_categorical_crossentropy(...)`
   Compute how wrong the model was.

5. `.backward()`
   Ask tinygrad to figure out how each trainable number should move.

6. `loss.realize(*opt.schedule_step())`
   Actually perform the work:
   compute the needed kernels and apply the optimizer update.

That last line is the trickiest one for beginners. Read it like this:

"Now really do the math, and also update the model weights."

### `get_test_acc`

This computes test accuracy.

It:

- runs the model on test images,
- picks the biggest score with `argmax`,
- compares that guess to the true label,
- computes the percentage correct.

### The main block

This part:

- loads MNIST,
- creates the model,
- chooses an optimizer,
- loops for some number of steps,
- periodically checks test accuracy,
- optionally fails if the final accuracy is too low.

That is the full tinygrad beginner workflow:

**load data -> build model -> train -> evaluate**

## Tinygrad-Specific "Magic" You Can Ignore At First

Three things in `beautiful_mnist.py` look more magical than they really are.

### `@function`

Think of this as:

"Please treat this math block as one reusable tinygrad function."

It helps tinygrad manage computation more cleanly.

You do not need to understand its implementation on day one.

### `@TinyJit`

Think of this as:

"If I do this same function many times, remember the fast path."

The first runs may capture the work. Later runs can replay it faster.

This is a speed feature, not a conceptual requirement for understanding training.

### `@Tensor.train()`

Think of this as:

"We are in training mode right now."

This matters for layers like BatchNorm that behave differently during training and evaluation.

## The Big Mental Model Of Tinygrad

Here is the whole project in one pipeline:

1. You write tensor code.
2. Tinygrad builds an internal graph.
3. Tinygrad groups that graph into runnable chunks.
4. Tinygrad generates low-level code for a backend.
5. Tinygrad asks a runtime to execute it on a device.
6. Results live in buffers until you ask for them.

That maps roughly to these files:

- `tinygrad/tensor.py`: user-facing tensor operations.
- `tinygrad/uop/ops.py`: internal graph representation.
- `tinygrad/engine/schedule.py`: grouping graph work into kernels.
- `tinygrad/codegen/`: turning work into lower-level program form.
- `tinygrad/renderer/`: emitting backend-specific program text.
- `tinygrad/engine/realize.py`: lowering and running work.
- `tinygrad/device.py`: choosing devices, buffers, compilers, allocators.
- `tinygrad/runtime/`: backend-specific device code.

If you get lost, come back to that list.

## The Important Objects To Know

### `Tensor`

The most important object in the whole repo.

This is:

- your data,
- intermediate activations,
- parameters,
- gradients,
- outputs.

If you are confused, ask: "what tensors exist right now, and what shape are they?"

### `nn`

This is the layer toolbox:

- `Linear`
- `Conv2d`
- `BatchNorm`
- optimizers
- datasets

Tinygrad keeps this intentionally small.

### `TinyJit`

This is the "remember a fast route" helper for repeated work.

### `Device`

This decides where work runs.

Examples:

- CPU
- Metal
- CUDA
- AMD
- WebGPU
- pure Python backend

### `UOp`

This is an internal operation node.

You can think of it as a lower-level instruction in tinygrad's private language.

Beginners do not need to manipulate these directly, but it helps to know they exist.

### `Buffer`

This is a piece of device memory.

Results do not become regular Python numbers immediately. They usually live in buffers until you call something like `.numpy()` or `.item()`.

## How To Use Tinygrad In Your Own Project

Use this progression.

### Level 1: Tiny tensor experiments

Start with plain tensor math.

Examples:

- vector addition,
- matrix multiply,
- softmax,
- tiny toy regression.

This teaches you the API without too much ML pressure.

### Level 2: Small custom model

Make a tiny model with:

- one or two `Linear` layers, or
- a small CNN like `beautiful_mnist.py`.

Use:

- `nn.Linear`
- `nn.Conv2d`
- `nn.optim.Adam`
- `nn.state.get_parameters`

### Level 3: Saving and loading

Once you can train a tiny model, learn:

- `tinygrad.nn.state.safe_save`
- `tinygrad.nn.state.safe_load`
- `tinygrad.nn.state.get_parameters`

That gets you much closer to real project use.

### Level 4: ONNX and existing models

If you want to connect tinygrad to outside model formats, look at:

- `tinygrad/nn/onnx.py`
- `tinygrad/nn/state.py`

Do this later, not first.

## How To Start Contributing

Do not start with a giant refactor.

Start small.

### Best first contribution types

- a missing edge-case test,
- a tiny bug fix,
- a regression test for a bug you found,
- a small docs clarification after you understand the code well.

### Best places to start reading

- `test/test_tiny.py`
- `test/null/`
- `test/unit/`
- `test/backend/`

These are easier entry points than deep runtime code.

### Good beginner contribution workflow

1. Reproduce one failing or confusing behavior.
2. Find the smallest test that covers it.
3. Run only that test.
4. Find the core file responsible.
5. Make one small change.
6. Rerun the same test.
7. Add a regression test if needed.

### Why tests matter so much here

In tinygrad, tests are one of the clearest maps of expected behavior.

The docs tell you what the project wants to be.
The tests tell you what the project must not break.

### Contribution norms from the repo

The main README makes the philosophy pretty clear:

- keep changes readable,
- do not code-golf,
- benchmark speed claims,
- keep PRs small,
- add regression tests,
- avoid touching advanced `extra/` code casually unless you know why.

## Hardware Advice For Beginners

You do not need to buy anything right away.

### Best zero-cost option

Use the machine you already have and start with:

```bash
PYTHONPATH=. PYTHON=1 python3 test/test_tiny.py TestTiny.test_plus
```

That is enough to start learning the library and making first contributions.

### Best easy real-hardware option

An Apple Silicon Mac with 16GB or more memory is a very beginner-friendly path.

Why:

- Metal support is first-class in tinygrad,
- setup is usually simpler than a custom Linux+GPU stack,
- you can get real acceleration without much ceremony.

### Best if you specifically want CUDA-style work

Use Linux with an NVIDIA GPU.

If buying for learning, prefer more VRAM over chasing the fanciest model name.

### What to avoid at first

Do not make your first week about:

- exotic hardware bring-up,
- low-level runtime internals,
- giant benchmark workflows,
- or `extra/` tools you do not need yet.

Learn the core library first.

## A Simple Reading Order For The Core

Use this sequence:

1. `docs/quickstart.md`
2. `examples/beautiful_mnist.py`
3. `test/test_tiny.py`
4. `tinygrad/tensor.py`
5. `tinygrad/nn/__init__.py`
6. `tinygrad/nn/optim.py`
7. `tinygrad/device.py`
8. `tinygrad/engine/schedule.py`
9. `tinygrad/engine/realize.py`
10. `tinygrad/uop/ops.py`

That order moves from "how to use it" to "how it really works".

## Tiny Exercises

Use these to learn and to teach others.

### Exercise 1

Change the batch size in `beautiful_mnist.py` using `BS=64`.

Question:
What changed? Only speed? Memory use? Accuracy?

### Exercise 2

Switch the optimizer from Adam to SGD with `SGD=1`.

Question:
What feels different about training progress?

### Exercise 3

Print tensor shapes inside the model.

Question:
Can you explain why the final linear layer uses `576` input features?

### Exercise 4

Read `test/test_tiny.py`.

Question:
Which tests feel like pure tensor behavior, and which feel closer to framework behavior?

### Exercise 5

Read `tinygrad/device.py`.

Question:
Where does tinygrad decide what backend to use?

## Five Questions To Ask While Reading Any File

When teaching others to read this repo, ask these every time:

1. What is this file responsible for?
2. Is this user-facing or internal?
3. What objects flow in and out of it?
4. What simpler file should I understand before this one?
5. If this file broke, which tests would probably fail?

These questions keep you from reading files as a giant wall of code.

## A Tiny Glossary

- **Activation**: the intermediate output of a layer.
- **Batch**: a small group of examples trained together.
- **Backend**: the device engine tinygrad runs on.
- **Gradient**: the direction a weight should move.
- **Inference**: running a trained model to make a guess.
- **JIT**: remembering and replaying a faster execution path.
- **Kernel**: a chunk of low-level work sent to a backend.
- **Loss**: how wrong the model is.
- **Optimizer**: the thing that updates weights.
- **Tensor**: a multi-dimensional box of numbers.
- **Training**: improving the model using examples and feedback.
- **UOp**: tinygrad's internal operation node.
- **Weight**: a learned parameter inside the model.

## What To Remember

If you forget everything else, remember this:

- tinygrad starts with `Tensor`.
- the best beginner example is `examples/beautiful_mnist.py`.
- tests are one of the best ways to learn the codebase.
- `tinygrad/` is the core, `extra/` is advanced.
- you can start learning and even contributing without buying a GPU.

That is enough to begin.
