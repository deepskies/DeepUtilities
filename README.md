# DeepBench

### What is it?
Pipeline for developing, training, and diagnosing machine learning algorithms
![Example Image of pipeline](https://github.com/deepskies/DeepUtilities/repository_support/images/DeepSkies_Logos_DeepUtilities.png) 


### Why do we need it? Why is it useful?
1. There are standard methods for pre-processing, training, and diagnosing algorithms; why should everyone reinvent the wheel every time?
2. Let's standardize and grow the library of diagnostics for science analyses with machine learning!


## General Features
1. modularity - replaceable/extensible pipeline elements
2. fully connected pipeline elements
3. good for both TensorFlow and PyTorch



## Requirements
1. python 3.x
2. more, but not sure yet.


## Example
![Example Image of pipeline](https://github.com/deepskies/DeepUtilities/repository_support/images/DeepUtilities_Schematic.png) 


## Pipeline Flow
1. Ingest
2. Pre-process
3. Train
4. Test
5. Diagnose


## Pipeline Elements
1. Ingest data
2. Data/Model API
3. Track Experiments


## Original Development Team
1. Praveen Balaji
2. Joao Caldeira
3. Callista Christ
4. Aleksandra Ciprijanovic
5. Anand Jain
6. Francois Lanusse
7. Brian Nord
8. Marwah Roussi
9. Amanda Whaley


## How to contribute
I'm really glad you're reading this, because we need volunteer developers to help this project come to fruition.


### Submitting changes

Please send a [GitHub Pull Request to DeepUtilities](https://github.com/deepskies/DeepUtilities/pull/new/master) with a clear list of what you've done (read more about [pull requests](http://help.github.com/pull-requests/)). When you send a pull request, we will love you forever if you include examples. We can always use more test coverage. Please follow our coding conventions (below) and make sure all of your commits are atomic (one feature per commit).

Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

    $ git commit -m "A brief summary of the commit
    > 
    > A paragraph describing what changed and its impact."

### Coding conventions

Start reading our code and you'll get the hang of it. We optimize for readability:

  * We indent using tabs
  * We ALWAYS put spaces after list items and method parameters (`[1, 2, 3]`, not `[1,2,3]`), around operators (`x += 1`, not `x+=1`), and around hash arrows.
  * This is open source software. Consider the people who will read your code, and make it look nice for them. It's sort of like driving a car: Perhaps you love doing donuts when you're alone, but with passengers the goal is to make the ride as smooth as possible.
