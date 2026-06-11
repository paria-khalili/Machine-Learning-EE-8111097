#!/usr/bin/env python
# coding: utf-8

#  **Question 5 — Genetic Algorithm: Jigsaw Puzzle Solver**
# 
# In this project, we use a Genetic Algorithm (GA) to automatically solve a jigsaw puzzle.  
# We rely on the open-source library **GAPS (Genetic Algorithm Puzzle Solver)**, available at:
#  https://github.com/nemanja-m/gaps
# 
# This notebook walks through:
# 
# 1. **(a) Fit Score** — defining how well two pieces fit  
# 2. **(b) Puzzle Fitness** — evaluating a whole puzzle arrangement  
# 3. **(c) Selection** — survival of the fittest  
# 4. **(d) Evolution Engine** — GA loop  
# 5. Running the GAPS solver  
# 6. Displaying the created puzzle and final solution  
# 

# **(a) Fit Score — Theory**
# 
# To judge how well **two puzzle pieces fit together**, we define a **fit score**.
# 
# A puzzle piece has 4 edges: **top, bottom, left, right**.  
# If two pieces are adjacent, their shared edges should visually match.
# 
# How It Works: 
# 
# **Difference Calculation**: The code subtracts the pixel values of the two edges, element by element, and squares the result. If the edges are a perfect match, the difference for every pixel is zero.
# 
# **Negative Sum**: It takes the negative sum of all these squared differences.
# 
# **Maximization Goal**:
#     Good Match: If the edges match closely, the squared difference sum is small (close to 0), making the fit_score close to $0$.
#     Bad Match: If the edges are very different, the squared difference sum is large, resulting in a large negative fit_score.
# 
# Since Genetic Algorithms are designed to maximize a fitness value, a score close to $0$ is the best possible score for any pair of edges, successfully steering the algorithm towards minimal pixel differences.
# 
# Why it works
# Puzzle edges that match in color and texture have:
# 
# - similar pixel patterns  
# - similar gradients  
# - similar brightness and edges  
# 
# Thus, their pixel difference is small.
# 
# This is exactly the mechanism used inside GAPS.
# 

# In[1]:


import os, sys, subprocess
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUTF8"] = "1"
from PIL import Image
import matplotlib.pyplot as plt


# In[11]:


import numpy as np

def fit_score(edge1, edge2):
    # pixel-wise difference
    diff = (edge1.astype('float') - edge2.astype('float')) ** 2
    return -np.sum(diff)


# **(b) Evaluate a Whole Puzzle — Theory**
# 
# Once we can measure how well **two pieces fit**, we compute **the fitness of an entire arrangement**.
# 
# A puzzle arrangement is a 2D grid. For each adjacent pair:
# 
# - compare **right edge** of piece `(i,j)` with **left edge** of `(i, j+1)`
# - compare **bottom edge** of piece `(i,j)` with **top edge** of `(i+1, j)`
# 
# The GA tries to **maximize** this total fitness, meaning the arrangement has many “good fits.”
# 
# This is exactly how GAPS evaluates candidate puzzles.
# 

# In[3]:


def puzzle_fitness(pieces, width, height):
    score = 0
    for i in range(height):
        for j in range(width):
            idx = i * width + j
            if j < width - 1:  # right neighbor
                score += fit_score(pieces[idx]["right"], pieces[idx+1]["left"])
            if i < height - 1:  # bottom neighbor
                score += fit_score(pieces[idx]["bottom"], pieces[idx+width]["top"])
    return score


# **(c) Survival of the Fittest — Theory**
# 
# Genetic Algorithms evolve a population of candidate puzzle arrangements.
# 
# We must choose which candidates become **parents**.  
# We use **roulette wheel selection**, proportional to fitness:
# 
# It calculates the Total Fitness of the entire population (total = sum(fitnesses)).
# 
# It imagines a roulette wheel where the size of each slice is proportional to the individual's fitness.
# It generates a random pick between $0$ and the total fitness.
# It then iterates through the individuals, summing their fitnesses (current += f), until the current sum exceeds the random pick. 
# The individual responsible for crossing this threshold is selected.
# 
# Result: A candidate solution with a higher fitness score takes up a larger slice of the wheel, making it more likely to be randomly chosen to be a parent. This is how the "fittest" individuals are selected to pass on their characteristics.
# 
# Thus:
# 
# - higher fitness → higher probability  
# - worse individuals → almost never chosen  
# 
# Why this works
# Good puzzle arrangements survive more, so the GA gradually improves.
# 

# In[4]:


import random

def roulette_selection(population, fitnesses):
    total = sum(fitnesses)
    pick = random.uniform(0, total)
    current = 0
    for individual, f in zip(population, fitnesses):
        current += f
        if current >= pick:
            return individual
    return population[-1]


# **(d) Evolution Engine — Theory**
# 
# A GA repeatedly applies the following cycle:
# 
# 1. **Initialization** — random puzzle arrangements  
# 2. **Selection** — choose parents based on fitness  
# 3. **Crossover** — combine pieces from two parents  
# 4. **Mutation** — random swaps to introduce diversity  
# 5. **Replacement** — build the next generation  
# 
# This loop repeats for a set number of generations (--generations=20 in your command):
# 
# Initialization: The first generation is created with random arrangements of all the puzzle pieces.
# 
# Evaluation: The puzzle_fitness function is used to calculate the score for every individual in the current population.
# 
# Selection: Two parents are chosen using the roulette_selection based on their high fitness.
# 
# Crossover: The algorithm combines sections from the two parents (e.g., taking a correctly matched block of pieces from Parent 1 and a different matched block from Parent 2) to create a child solution. This propagates "good" arrangements.
# 
# Mutation: A small chance of randomly swapping two pieces in the child is introduced. This prevents the algorithm from getting stuck in a local optimum and ensures new arrangements are explored.
# 
# Replacement: The children form the next_population, replacing the old one.
# 
# By repeating this cycle, each generation becomes slightly "fitter" than the last, until the population converges on the arrangement with the highest possible fitness score—the solved puzzle.
# 

# In[5]:


def evolve():
    population = init_population()
    for gen in range(100):
        fitnesses = [puzzle_fitness(ind) for ind in population]
        next_population = []
        for _ in range(len(population)):
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
            child = crossover(parent1, parent2)
            mutate(child)
            next_population.append(child)
        population = next_population


# In[6]:


image_path = r"C:\Users\waran\gaps\images\pillars.jpg"
puzzle_out = r"C:\Users\waran\gaps\puzzle.jpg"
solution_out = r"C:\Users\waran\gaps\solution.jpg"
piece_size = 64


# In[7]:


cmd = ['gaps', 'create', image_path, puzzle_out, f'--size={piece_size}']
print("Running:", " ".join(cmd))

result = subprocess.run(cmd)
print("Create return code:", result.returncode)


# In[8]:


if os.path.exists(puzzle_out):
    img = Image.open(puzzle_out)
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Created Puzzle (shuffled pieces)")
else:
    print("Puzzle image not found:", puzzle_out)


# In[9]:


import subprocess

cmd = ['gaps', 'run', puzzle_out, solution_out,
       '--generations=20', '--population=600',
       f'--size={piece_size}', '--debug']

print("Running:", " ".join(cmd))

# Safe subprocess call that captures stdout/stderr
result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True  # decode output as text
)

print("Return code:", result.returncode)
print("\n--- stdout ---\n", result.stdout)
print("\n--- stderr ---\n", result.stderr)



# In[10]:


import os
from PIL import Image
import matplotlib.pyplot as plt

# The code to display the solved puzzle image
if os.path.exists(solution_out):
    img_solution = Image.open(solution_out)
    plt.figure(figsize=(6, 6))
    plt.imshow(img_solution)
    plt.axis("off")
    plt.title("GA Solved Puzzle")
    plt.show()
else:
    print(f"Solution image not found at: {solution_out}. Check the output of the 'gaps run' command.")


# **Question 14: Step-Size Strategies in Gradient Descent**
# 

# **(a)**: 
# 

# **Step-Size Selection Methods:**
# Gradient Descent requires choosing a step-size (learning rate) $\alpha$. Two widely used adaptive strategies are designed to find an $\alpha$ that provides good progress without overshooting the minimum.

# **Armijo Backtracking Line Search**
# 
# Armijo Backtracking is the simpler and more robust method. It starts with a guess ($\alpha=1$) and repeatedly shrinks it until a sufficient decrease in the function value is achieved.
# 
# The step-size $\alpha$ must satisfy the Armijo condition:
# 
# $$f(x - \alpha \nabla f(x)) \le f(x) - c_1 \alpha \|\nabla f(x)\|^2,$$
# 
# where:
# $c_1 \in (0,1)$ is a small constant (e.g., $10^{-4}$). This term ensures the function decrease is not too small relative to the gradient size.
# 
# The method repeatedly tests the condition, shrinking $\alpha \leftarrow \beta \alpha$ (with $\beta \in (0,1)$, e.g., $\beta=0.5$) until the inequality holds.
# 
# Effect: Armijo ensures stability and prevents the algorithm from diverging (overshooting). It is robust but can sometimes select a conservatively small step-size, leading to slow convergence.

# **Wolfe Conditions**
# 
# The Wolfe conditions are a stricter set of requirements designed to select a more optimal step size by bounding $\alpha$ from both above (Armijo) and below (Curvature).
# 
# It requires two inequalities:
# 
# 1. Armijo Condition (Sufficient Decrease)
# 
# This is the same condition as above, ensuring the step yields a significant reduction in $f(x)$.
# 
# $$f(x - \alpha \nabla f(x)) \le f(x) - c_1 \alpha \|\nabla f(x)\|^2$$
# 
# 2. Curvature Condition (Prevents Trivial Steps)
# 
# This condition ensures the step is not too small by requiring the slope (gradient) at the new point, $x + \alpha p$, to be significantly flatter than the slope at the current point, $x$.
# 
# $$|\nabla f(x - \alpha \nabla f(x))^\top (-\nabla f(x))| \le c_2 \|\nabla f(x)\|^2,$$
# 
# where $c_2 \in (c_1,1)$, typically $0.9$. The term on the left is the directional derivative along the search direction $p = -\nabla f(x)$.
# 
# Effect: Wolfe tends to select larger, more optimal step sizes than Armijo alone, often resulting in faster convergence.
# 

# **(b):**

# Effect: Wolfe tends to select larger, more optimal step sizes than Armijo alone, often resulting in faster convergence.
# 
# $$f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2.$$
# 
# Gradient:
# 
# $$\nabla f(x) = 
# \begin{bmatrix}
# -400x_1(x_2 - x_1^2) - 2(1-x_1) \\
# 200(x_2 - x_1^2)
# \end{bmatrix}$$
# 
# We run Gradient Descent (GD) starting at the point:
# 
# $$x_0 = [-1.2, 1]^T.$$

# **Gradient Descent Implementation**
# 
# The main driver function gradient_descent implements the core optimization loop:
# 
# 
# 1.Calculate the negative gradient: $p = -\nabla f(x)$.
# 
# 
# 2.Use the specified step_rule (armijo or wolfe) to find the optimal step-size $\alpha$.
# 
# 3.Update the position: $x \leftarrow x + \alpha p$.
# 
# 4.Repeat until the gradient is near zero or the maximum iterations are reached.
# 
# The code runs Gradient Descent using the Rosenbrock function with the starting point $x_0$:
# 
# GD + Armijo: Uses the robust, sufficient decrease rule.
# 
# GD + Wolfe: Uses the stricter two-condition rule.
# 
# The results (traj_armijo, f_armijo, etc.) store the path and function value history for comparison.

# In[12]:


import numpy as np
import matplotlib.pyplot as plt


# In[15]:


def f(x):
    x1, x2 = x
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def grad(x):
    x1, x2 = x
    df1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
    df2 = 200*(x2 - x1**2)
    return np.array([df1, df2])


# In[16]:


def armijo_backtracking(x, p, c1=1e-4, beta=0.5):
    alpha = 1.0
    fx = f(x)
    gradx = grad(x)

    while f(x + alpha*p) > fx + c1 * alpha * gradx @ p:
        alpha *= beta
        
    return alpha


# In[17]:


def wolfe_line_search(x, p, c1=1e-4, c2=0.9):
    alpha = 1.0
    fx = f(x)
    gradx = grad(x)
    gradp = gradx @ p

    while True:
        # Armijo
        if f(x + alpha*p) > fx + c1 * alpha * gradp:
            alpha *= 0.5
        # Curvature
        elif abs(grad(x + alpha*p) @ p) > c2 * abs(gradp):
            alpha *= 0.5
        else:
            break
            
    return alpha


# In[18]:


def gradient_descent(x0, step_rule, max_iter=200):
    x = x0.copy()
    hist = [x.copy()]
    fvals = [f(x)]

    for k in range(max_iter):
        g = grad(x)
        p = -g

        # choose step size rule
        if step_rule == "armijo":
            alpha = armijo_backtracking(x, p)
        elif step_rule == "wolfe":
            alpha = wolfe_line_search(x, p)
        else:
            raise ValueError("Unknown step rule")

        x = x + alpha * p

        hist.append(x.copy())
        fvals.append(f(x))

        if np.linalg.norm(g) < 1e-6:
            break
    
    return np.array(hist), np.array(fvals)


# In[19]:


x0 = np.array([-1.2, 1.0])

traj_armijo, f_armijo = gradient_descent(x0, "armijo")
traj_wolfe, f_wolfe = gradient_descent(x0, "wolfe")


# In[20]:


# contour plot
x1 = np.linspace(-2, 2, 400)
x2 = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = 100*(X2 - X1**2)**2 + (1 - X1)**2

plt.figure(figsize=(8,6))
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20))

plt.plot(traj_armijo[:,0], traj_armijo[:,1], 'o-', label="Armijo")
plt.plot(traj_wolfe[:,0], traj_wolfe[:,1], 's-', label="Wolfe")

plt.plot([1], [1], 'rx', markersize=12, label="Minimizer")

plt.legend()
plt.title("Trajectories of GD on Rosenbrock Function")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


# In[21]:


plt.figure(figsize=(7,5))

plt.plot(np.log10(f_armijo), label="Armijo")
plt.plot(np.log10(f_wolfe), label="Wolfe")

plt.xlabel("Iteration")
plt.ylabel("log10 f(x)")
plt.title("Convergence Comparison")
plt.legend()
plt.show()


# **(c):**

# Trajectories on Rosenbrock Contours:
# This plot shows the path each method takes across the contour lines of the Rosenbrock function.
# 
# Armijo: The path is stable but often takes many small, zig-zagging steps, particularly along the narrow, curved valley leading to the minimum at $(1, 1)$.
# 
# Wolfe: The path is smoother and covers more distance in the direction of the minimum. It successfully adapts to the curvature of the valley, selecting larger $\alpha$ values when appropriate.

# **Function Value vs Iteration** (log scale)
# 
# This plot is the key measure of convergence speed.The vertical axis is $\log_{10}(f(x))$, which emphasizes large changes and linearizes convergence behavior.
# 
# The plot clearly shows that Wolfe converges significantly faster (reaches a lower function value in fewer iterations) than Armijo.
# 
# **Convergence Behavior**
# 
# Armijo:
# 
# Simple and Robust: Provides stable convergence by strictly enforcing sufficient function decrease.
# 
# Slower Convergence: Since it only checks for decrease and doesn't explicitly prevent steps from being too small, it often selects conservative $\alpha$ values, particularly in difficult problems like the Rosenbrock valley.
# 
# Wolfe Conditions:
# 
# Faster Convergence: By adding the Curvature Condition, the method ensures the chosen $\alpha$ is not trivially small, maximizing the progress in each step.
# 
# More Optimal Steps: It finds a better balance between sufficient decrease and large step size.
# 
# Conclusion:
# 
# For non-trivial optimization problems, the Wolfe line search provides:faster convergence (fewer iterations needed), a smoother trajectory, and a more efficient overall solution.
# 
# While Armijo is easier to implement and guarantees stability, the performance gain from using Wolfe often justifies the added complexity.

# 
# **Question 13: (c) (Coding) Write the 3 ×3 nonlinear system F(w) = 0 with w = [x,y,λ]⊤**

# In[4]:


import numpy as np

def F(w):
    x, y, lam = w
    exp_xy = np.exp(x**2 + y**2)
    F1 = 2*x*exp_xy + lam      # ∂L/∂x
    F2 = 2*y*exp_xy + lam      # ∂L/∂y
    F3 = x + y - 1             # constraint
    return np.array([F1, F2, F3])

# ---- produce output ----
w0 = np.array([0.0, 0.0, 0.0])   # example input
F(w0)


# **Question 12: Rosenbrock Function: GD vs. Newton (Optional)**
# 
# Comparing Steepest Descent (GD) and Pure Newton's Method
# The Rosenbrock function is $f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1 - x_1)^2$.
# 
# Gradient:
# 
# $\nabla f(x)$We take the partial derivatives with respect to $x_1$ and $x_2$:$$\nabla f(x) = 
# \begin{bmatrix}
# \frac{\partial f}{\partial x_1} \\
# \frac{\partial f}{\partial x_2}
# \end{bmatrix}
# =
# \begin{bmatrix}
# -400x_1(x_2 - x_1^2) - 2(1-x_1) \\
# 200(x_2 - x_1^2)
# \end{bmatrix}
# $### Hessian $H(x)$
# 
# The Hessian matrix contains the second-order partial derivatives:
# 
# $$H(x) =
# \begin{bmatrix}
# \frac{\partial^2 f}{\partial x\_1^2} & \frac{\partial^2 f}{\partial x\_1 \partial x\_2} \\
# \frac{\partial^2 f}{\partial x\_2 \partial x\_1} & \frac{\partial^2 f}{\partial x\_2^2}
# \end{bmatrix}
# $$The components are:
# 
#   * $\frac{\partial^2 f}{\partial x_1^2} = 1200 x_1^2 - 400 x_2 + 2$
#   * $\frac{\partial^2 f}{\partial x_2^2} = 200$
#   * $\frac{\partial^2 f}{\partial x_1 \partial x_2} = \frac{\partial^2 f}{\partial x_2 \partial x_1} = -400 x_1$
# 
# Thus:
# 
# $$H(x) = 
# \begin{bmatrix}
# \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} \\
# \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2}
# \end{bmatrix}
# =
# \begin{bmatrix}
# 1200 x_1^2 - 400 x_2 + 2 & -400 x_1 \\
# -400 x_1 & 200
# \end{bmatrix}$$

# In[7]:


import numpy as np
import matplotlib.pyplot as plt

# --- Optimization Parameters ---
x0 = np.array([-1.2, 1.0]) # Starting point
MAX_ITER = 50              # Maximum iterations for comparison
TOL = 1e-6                 # Tolerance for gradient norm (stopping criterion)

# --- Rosenbrock Function, Gradient, and Hessian ---

def f(x):
    """Rosenbrock function value."""
    x1, x2 = x
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

def grad(x):
    """Rosenbrock gradient ∇f(x)."""
    x1, x2 = x
    df1 = -400*x1*(x2 - x1**2) - 2*(1 - x1)
    df2 = 200*(x2 - x1**2)
    return np.array([df1, df2])

def hess(x):
    """Rosenbrock Hessian H(x)."""
    x1, x2 = x
    h11 = 1200*x1**2 - 400*x2 + 2
    h12 = -400*x1
    h22 = 200
    return np.array([[h11, h12], [h12, h22]])


# **(b) & (c) Implementation of Steepest Descent and Pure Newton's Method**
# 
# 1. Steepest Descent (GD)
# 
# Search Direction: $p_k = -\nabla f(x_k)$ (opposite of the gradient).
# 
# Step Size: Fixed $\alpha = 0.001$.
# 
# Update: $x_{k+1} = x_k + \alpha p_k$
# 
# 2. Pure Newton's Method
# 
# Search Direction: $p_k = -H(x_k)^{-1} \nabla f(x_k)$ (uses the Hessian to find the minimum of a quadratic approximation).
# 
# Step Size: Full step $\alpha = 1.0$.
# 
# Update: $x_{k+1} = x_k + \alpha p_k$

# In[8]:


def steepest_descent(x0, alpha=0.001, max_iter=MAX_ITER):
    """Steepest Descent (GD) with fixed step size."""
    x = x0.copy()
    hist = [x.copy()]
    fvals = [f(x)]

    for k in range(max_iter):
        g = grad(x)
        
        # Check convergence
        if np.linalg.norm(g) < TOL:
            break
            
        p = -g  # Steepest Descent direction
        x = x + alpha * p
        
        hist.append(x.copy())
        fvals.append(f(x))
    
    return np.array(hist), np.array(fvals), k + 1

def pure_newton(x0, alpha=1.0, max_iter=MAX_ITER):
    """Pure Newton's Method with full steps (α=1)."""
    x = x0.copy()
    hist = [x.copy()]
    fvals = [f(x)]

    for k in range(max_iter):
        g = grad(x)
        H = hess(x)
        
        # Check convergence
        if np.linalg.norm(g) < TOL:
            break
            
        try:
            # Solve H * p = -g for the Newton direction p
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            print(f"Hessian singular at iteration {k}. Switching to GD direction.")
            p = -g # Fallback to GD direction if Hessian is singular
        
        x = x + alpha * p
        
        hist.append(x.copy())
        fvals.append(f(x))
    
    return np.array(hist), np.array(fvals), k + 1


# **(d) Run Optimization**

# In[9]:


# Run Steepest Descent (GD)
traj_gd, f_gd, iter_gd = steepest_descent(x0)
print(f"Steepest Descent finished in {iter_gd} iterations. Final f(x): {f_gd[-1]:.2e}")

# Run Pure Newton's Method
traj_newton, f_newton, iter_newton = pure_newton(x0)
print(f"Newton's Method finished in {iter_newton} iterations. Final f(x): {f_newton[-1]:.2e}")


# **(e) Plotting and Discussion**
# 
# (i) Trajectories on a Contour Plot
# 
# The contour plot visually demonstrates the difference in the search directions and step sizes between the two methods.

# In[10]:


# --- Setup Contour Plot ---
x1_range = np.linspace(-2, 2, 400)
x2_range = np.linspace(-1, 3, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
Z = 100*(X2 - X1**2)**2 + (1 - X1)**2

plt.figure(figsize=(10, 8))
# Plot log-spaced contours to highlight the narrow valley
plt.contour(X1, X2, Z, levels=np.logspace(-1, 3, 20), cmap='viridis') 

# --- Plot Trajectories ---
plt.plot(traj_gd[:, 0], traj_gd[:, 1], 'o-', markersize=4, label=f"Steepest Descent (k={iter_gd})", alpha=0.7)
plt.plot(traj_newton[:, 0], traj_newton[:, 1], 's-', markersize=6, label=f"Newton's Method (k={iter_newton})", alpha=0.9)

# Plot minimizer and start point
plt.plot([1], [1], 'rx', markersize=12, label="Minimizer $[1, 1]^T$")
plt.plot(x0[0], x0[1], 'ko', markersize=8, label="Start $x_0$")

plt.title("Trajectories of GD vs. Newton on Rosenbrock Function")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.xlim(-2, 2)
plt.ylim(-1, 3)
plt.legend()
plt.grid(True)
plt.show()


# (ii) $\log f(x_k)$ vs. Iteration
# 
# This plot directly compares the convergence rate of the two algorithms.

# In[11]:


plt.figure(figsize=(8, 6))

plt.plot(np.log10(f_gd), label="Steepest Descent ($\\alpha=0.001$)")
plt.plot(np.log10(f_newton), label="Pure Newton ($\\alpha=1.0$)")

plt.axhline(np.log10(TOL), color='r', linestyle='--', label=f'Tolerance ($10^{{{np.log10(TOL):.0f}}}$)')

plt.title("Convergence Comparison: log10 f($x_k$) vs. Iteration")
plt.xlabel("Iteration (k)")
plt.ylabel("log$_{10}$ f($x_k$)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.ylim(-6, 4)
plt.show()


# **Convergence Rates**
# 
# Steepest Descent (GD):
# 
# The GD method exhibits linear convergence.
# 
# 
# - Trajectory: The plot shows severe zig-zagging because the gradient direction is often nearly orthogonal to the direction of the minimum in the narrow, curved valley.
# 
# 
# - Performance: The fixed, small step size ($\alpha=0.001$) is necessary for stability but causes the progress to be very slow. It requires a large number of steps to reach the tolerance, making it inefficient for this problem.
# 
# Pure Newton's Method:
# 
# The Newton's method exhibits quadratic convergence.
# 
# - Trajectory: The path is short and direct. Newton's method uses the Hessian (second-order derivative) to model the function's curvature, allowing it to calculate a step that goes nearly straight to the minimum of the quadratic approximation.
# 
# - Performance: The convergence plot shows a dramatic drop in $f(x)$ after the first few iterations. It reaches the specified tolerance in a tiny fraction of the iterations required by GD.
# 
# 
# 

# **Conclusion**
# 
# **1. The Core Problem: The Rosenbrock "Banana Valley"**
# 
# The Rosenbrock function is often called the "banana function" because of its highly eccentric, narrow, and curved minimum valley.
# 
# - Poor Scaling: As you move into the valley, the function slopes are very steep across the valley (perpendicular to the minimum) but very shallow along the valley floor (parallel to the minimum).
# 
# - The Problem: This huge difference in steepness is called poor conditioning. It means the landscape is highly distorted, which severely hurts optimization methods that rely only on the local gradient.
# 
# **2. Steepest Descent (GD): The "Myopic Hiker"** 
# 
# Steepest Descent is a first-order method—it only uses the gradient ($\nabla f(x)$), which represents the direction of maximum immediate descent.
# 
# - Limited Knowledge: GD is like a hiker who can only see the tip of their boot. They always take a step in the steepest direction.
# 
# - Failure in the Valley: In the narrow Rosenbrock valley, the direction of the steepest descent is almost always perpendicular to the valley floor, not along it.
# 
# - The Result: Zig-Zagging: GD is forced to take tiny steps, repeatedly bouncing back and forth across the valley, making very little progress toward the minimum. This slow, predictable progress is called linear convergence. The number of correct digits in the solution increases only by an additive constant with each iteration.

# In[ ]:




