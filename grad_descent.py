import numpy as np
from sympy import symbols, diff, solve


class GradDescent():
    def __init__(self, func, vars):
        '''Arguments required are a func: a function of variables given by vars using sympy symbols.'''
        self.func = func
        self.vars = vars
        self.beta = 0.1
    def step(self, step_size, method):
        if method == "fixed":
            return step_size
        elif method == "exact":
            t = symbols("t")
            expr = np.array(self.point) + t * self.direction(self.partial_diffs(self.func), self.point)
            sub = [(self.vars[i], f) for i, f in enumerate(expr)]
            ft = self.func.subs(sub)
            step = solve(diff(ft, t), t)
            return np.array(step) * (-1)
        elif method == "backtracking":
            return step_size * self.beta
        elif method == "newton":
            hessian = np.array(self.second_deriv(self.func))
            try:
                return np.linalg.inv(hessian.astype(float))
            except TypeError:
                #Solving a 2x2 matrix specifically for problem 3 in the case of passing variables
                det = hessian[0][0]*hessian[1][1] - hessian[0][1]*hessian[1][0]
                a = hessian[1][1] / det
                b = - hessian[0][1] /det
                c = - hessian[1][0] /det
                d = hessian[0][0] / det
                return np.array([[a, b], [c, d]])
        else:
            print("Incorrect step method chosen, default to fixed step")
            return step_size
    
    def second_deriv(self, func):
        '''return a list of second partial derivatives given a function with respect to self.vars'''
        partials = [self.partial_diffs(d) for d in self.partial_diffs(self.func)]
        return partials

    def partial_diffs(self, func):
        '''return a list of partial derivatives with respect to class list self.vars'''
        d = [diff(func, x) for x in self.vars]
        return d
    def direction(self, funcs, point):
        '''evaluates a list of functions at given point'''
        evals = [(x, point[i]) for i, x in enumerate(self.vars)]
        direction = [f.subs(evals) for f in funcs]
        return np.array(direction)
    def take_steps(self, steps, start_point, step_size = 0.9, method = "fixed", beta = 0.1):
        '''Step_size must be a float. Method can be 
        'fixed', 'exact', 'backtracking', or 'newton'. Default is 'fixed'.  
        start_point must be a list of floats or variables'''
        self.point = np.array(start_point)
        self.beta = beta
        self.point_log = [start_point]
        print(f"Starting at: {self.point}")
        for i in range(steps):
            if method == "newton":
                newx = self.point - np.matmul(self.step(step_size, method), self.direction(self.partial_diffs(self.func), self.point))
            else:
                newx = self.point - self.step(step_size, method) * self.direction(self.partial_diffs(self.func), self.point)
            self.point_log.append(newx)
            self.point = newx
            print(f"Step {i+1}: {newx}")
    def netwon_condition(self, error):
        condi = sum(self.direction(self.func, self.point)) / sum(self.direction(self.second_deriv(self.func)))
        return condi < error
    def backtrack_condition(self, alpha = 0.5):
        pass


x1, x2 = symbols('x1 x2')

f1 = x1**2 - 2*x1*x2 + 2*x2**2 + 2*x1
vars1 = [x1, x2]
start1 = [5.,5.]
p1 = GradDescent(f1, vars1)


f2 = x1**2 - 2*x1*x2 + 2*x2**2 - 4*x1 - 6*x2
start2 = [100.,100.]
p2 = GradDescent(f2, vars1)

b = symbols('b')
start3 = [b, b]
f3 = x1**4 + 2*x1**2*x2**2 + x2**4
p3 = GradDescent(f3, vars1)

p1.take_steps(20, start1, step_size = 1.2, method = "exact")
p2.take_steps(20, start2, step_size = 0.9, method = "newton")
p3.take_steps(1, start3, step_size = 0.9, method = "newton")