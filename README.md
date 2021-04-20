# Enhancing Restricted Boltzmann Machines Reconstructability Through Meta-Heuristic Optimization

*This repository holds all the necessary code to run the very-same experiments described in the paper "Enhancing Restricted Boltzmann Machines Reconstructability Through Meta-Heuristic Optimization".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `utils`
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `objects.py`: Wraps objects instantiation for command line usage;
   * `optimizer.py`: Wraps the optimization task into a single method;  
   * `target.py`: Implements the objective functions to be optimized.
   
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

In order to run the experiments, you can use `torchvision` to load pre-implemented datasets.

---

## Usage

### Model Training

The first step is to pre-train an RBM architecture. To accomplish such a step, one needs to use the following script:

```Python
python rbm_training.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Model Optimization

After conducting the training task, one needs to optimize the weights over the validation set. Please, use the following script to accomplish such a procedure:

```Python
python rbm_optimization.py -h
```

### Model Evaluation

Finally, it is now possible to evaluate a model using the testing set. Please, use the following script to accomplish such a procedure:

```Python
python rbm_evaluation.py -h
```

### Analyze Optimization Convergence (Optional)

Additionally, one can gather the optimization history files and input them to a script that analyzes its convergence and produces a plot that compares how each optimization technique has performed during its procedure. Please, use such a script as follows:

```Python
python analyze_optimization_convergence.py -h
```

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---
