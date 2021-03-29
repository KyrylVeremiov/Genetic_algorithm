import numpy as np

class ga:
    # %%
    D = 2
    size_of_population = 20
    tournament_size = 2  # size_of_population%tournament_size==0
    t=10

    A = -10
    B = 10
    f=None
    # def f(X):
    #     return abs(X * np.sin(X) + 0.1 * X).sum()

    n_bits = 4



    #%%
    def int_to_bytes(P):
        A = np.array(
            [['+' + bin(P[j][i])[2:] if P[j][i] >= 0 else '-' + bin(P[j][i])[3:] for i in range(P.shape[1])] for j in
             range(P.shape[0])])
        return np.array(
            [[A[j][i][0] + str('0' * (ga.n_bits - len(A[j][i][1:]))) + A[j][i][1:] for i in range(A.shape[1])] for j in
             range(A.shape[0])])


    def bytes_to_int(P):
        return np.array([[(-1 if P[j][i][0] == '-' else 1) * int('0b' + P[j][i][1:], 2) for i in range(P.shape[1])] for j in
                         range(P.shape[0])])


    # %%
    def fitness(P):
        X = ga.bytes_to_int(P)
        return np.array([(P[i], ga.f(X[i])) for i in range(X.shape[0])],dtype=object)

    #%%

    def get_population(P):
        # #%%
        # P=fit_population[:,0]
        newP=np.array([list(P[i]) for i in range(P.shape[0])])
        X = ga.bytes_to_int(newP)
        return np.array([(X[i], ga.f(X[i])) for i in range(X.shape[0])],dtype=object)

    def get_population_condition(X):
        return (abs(X[:, 1]).sum()) / X.shape[0]


    # %%
    def make_initial_population(count, a, b):
        # >>> random.sample(range(1, 100), 3)
        return ga.int_to_bytes(np.random.randint(a, b, (count, ga.D)))


    # %%
    def tournament_selection(fP, n,sizeP):
        newP = []
        for i in range(n):
            np.random.shuffle(fP)
            #for exponential population
            # G = np.array([fP[d:d + n] for d in range(0, fP.shape[0], n)])
            G = np.array([fP[d:d + n] for d in range(0, sizeP, n)])

            # print(G)
            newPi = [min(G[i], key=lambda item: item[1]) for i in range(G.shape[0])]
            newP += newPi
            # print(newP)
            # print(len(newP))
            # print('------')
        return np.array(newP)




    # %%
    def crossover(p1, p2):
        # #%%
        # p1=fit_population[0]
        # p2=fit_population[1]
        # print(p1,p2)

        c1 = []
        c2 = []
        for i in range(ga.D):
            g1 = p1[0][i]
            g2 = p2[0][i]
            # print(g1, g2)
            ng1i = g1[:int(ga.n_bits / 2) + 1] + g2[int(ga.n_bits / 2) + 1:]
            ng2i = g2[:int(ga.n_bits / 2) + 1] + g1[int(ga.n_bits / 2) + 1:]
            c1.append(ng1i)
            c2.append(ng2i)

        return [np.array(c1), np.array(c2)]


    def crossover_population(P, count_of_children):
        new_children = []
        while len(new_children) < count_of_children:
            new_children += ga.crossover(P[np.random.randint(0, P.shape[0])], P[np.random.randint(0, P.shape[0])])
        return np.array(new_children)


    # %%
    def mutate(p):
        # # %%
        # p = fit_population[1]
        # print(p)

        new_p=[]
        for i in range(ga.D):
            g = p[i]
            I = np.random.randint(1, ga.n_bits+1)
            # print(I)
            newg=g[:I]+str(0 if g[I] == '1' else 1)+g[I+1:]
            new_p.append(newg)
        return [new_p]

    def mutate_population(P):
        new_generation = []
        for i in range(P.shape[0]):
            new_generation += ga.mutate(P[i])
        return np.array(new_generation)

    def show_results(fP):
        tmp_population = ga.get_population(fP[:,0])
        print()
        print('X: ')
        print(min(tmp_population, key=lambda item: item[1]))
        print('Population:')
        print(tmp_population)
        print('Total score:')
        print(ga.get_population_condition(tmp_population))
        print('--------------------------------------')
        print()

    #%%
    def solve(F,show_calculations):
        ga.f=F
        # %%
        population = ga.make_initial_population(ga.size_of_population, ga.A, ga.B)
        # %%
        fit_population = ga.fitness(population)
        # print(fit_population)

        #%%
        if show_calculations:
            print('Initial: ')
            ga.show_results(fit_population)
            # tmp_population=get_population()
            # print('X: ')
            # print(min(fit_population, key=lambda item: item[1]))
            # print('Population:')
            # print(fit_population)
            # print('Total score:')
            # print(get_population_condition(fit_population))
            # print('--------------------------------------')
            # print()
        #%%
        for i in range(ga.t):
            new_children = ga.crossover_population(fit_population, ga.size_of_population)
            new_generation=ga.mutate_population(new_children)
            new_fit_generaton=ga.fitness(new_generation)
            fit_population = ga.tournament_selection(np.concatenate((new_fit_generaton,fit_population)), ga.tournament_size, ga.size_of_population)

            if show_calculations:
                # new_population=get_population(fit_population[:,0])
                print('Iteration: ',i)
                ga.show_results(fit_population)
                # print()
                # print('--------------------------------------')
                # print('X: ')
                # print(min(new_population, key=lambda item: item[1]))
                # print('Population:')
                # print(new_population)
                # print('Total score:')
                # print(get_population_condition(new_population))
                # print('--------------------------------------')
                # print()

        final_population=ga.get_population(fit_population[:,0])

        X=min(final_population, key=lambda item: item[1])
        if show_calculations:
            print('Final: ')
            ga.show_results(fit_population)
            #%%
            # print('X: ')
            # print(X)
            # print('Population:')
            # print(final_population)
            # print('Total score:')
            # print(get_population_condition(final_population))
        # return X

def F(X):
        return abs(X * np.sin(X) + 0.1 * X).sum()

ga.solve(F,True)

import matplotlib.pyplot as plt

def f1(x):
    return x * np.sin(x) + 0.1 * x

X=np.arange(-10, 10, 0.1)
plt.plot(X,f1(X))
plt.show()