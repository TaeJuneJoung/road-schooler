import random, copy
#import other class
from network import Network
class Generation():
    def __init__(self):

        self.genomes = []
        self.population = 64
        self.keep_best = 16
        #lucky_few is what?
        self.lucky_few = 16
        #chance_of_mutation is what?
        #돌연변이 발생률?
        self.chance_of_mutation = 0.1

    def set_initial_genomes(self):
        genomes = []
        for i in range(self.population):
            genomes.append(Network())
        return genomes

    def keep_best_genomes(self):
        print(self.genomes)
        self.genomes.sort(key=lambda i:i.fitness, reverse=True)
        print(self.genomes)
        #우성인자를 뽑아서 저장함
        self.best_genomes = self.genomes[:self.keep_best]

    def mutations(self):
        while len(self.genomes) < self.keep_best * 3:
            genome1 = random.choice(self.best_genomes)
            genome2 = random.choice(self.best_genomes)
            #mutate, cross_over는 아래의 함수
            self.genomes.append(self.mutate(self.cross_over(genome1, genome2)))
    
        while len(self.genomes) < self.population:
            genome = random.choice(self.best_genomes)
            self.genomes.append(self.mutate(genome))
        
        random.shuffle(self.genomes)

        return self.genomes

    def cross_over(self, genome1, genome2):
        new_genome = copy.deepcopy(genome1)
        other_genome = copy.deepcopy(genome2)

        cut_location = int(len(new_genome.w1) * random.uniform(0, 1))
        print('cut_location :',cut_location)
        for i in range(cut_location):
            new_genome.w1[i], other_genome.w1[i] = other_genome.w1[i], new_genome.w1[i]
        
        cut_location = int(len(new_genome.w2) * random.uniform(0, 1))
        for i in range(cut_location):
            new_genome.w2[i], other_genome.w2[i] = other_genome.w2[i], new_genome.w2[i]
        
        cut_location = int(len(new_genome.w3) * random.uniform(0, 1))
        for i in range(cut_location):
            new_genome.w3[i], other_genome.w3[i] = other_genome.w3[i], new_genome.w3[i]
        
        return new_genome

    def mutate_weights(self, weights):
        if random.uniform(0, 1) < self.chance_of_mutation:
            return weights * (random.uniform(0, 1) - 0.5) * 3 + (random.uniform(0, 1) - 0.5)
        else:
            return 0
    
    def mutate(self, genome):
        new_genome = copy.deepcopy(genome)
        new_genome.w1 += self.mutate_weights(new_genome.w1)
        new_genome.w2 += self.mutate_weights(new_genome.w2)
        new_genome.w3 += self.mutate_weights(new_genome.w3)
        return new_genome