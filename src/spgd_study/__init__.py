"""SPGD Empirical Study — package root.

Contents (filled in step-by-step; see instructions/step_XX_*.txt):
    optimizers/  : SPGD, PGD, RPGD optimizer implementations  (Step 1)
    benchmarks   : Rastrigin, Ackley, Rosenbrock              (Step 2)
    diagnostics  : stagnation episodes, escape time           (Step 1)
    models       : MLPs, ResNet-18 wrapper                    (Step 3+)
    data         : Two Moons / OpenML / CIFAR-10 loaders      (Step 3+)
    runner       : generic train loop with diagnostics        (Step 2)
    viz          : trajectory + landscape plots               (Step 2)
    utils        : seeds, logging, CSV I/O                    (Step 1)
"""

__version__ = "0.1.0"
