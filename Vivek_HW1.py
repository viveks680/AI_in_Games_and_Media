import math
import operator
from functools import reduce
import random

from PIL import Image, ImageDraw, ImageChops
from numpy.core._multiarray_umath import square
from numpy.ma import sqrt



class gene(object):
    """
    The gene class, which represent a triangle.
    Attributes:
        pos_1: The first vertex of the triangle.
        pos_2: The second vertex of the triangle.
        pos_2: The third vertex of the triangle.
        color: The GRBA value of the triangle.
    """
    def __init__(self):
        """
        Inits a gene with mutation operation.
        """
        self.mutate()

    def mutate(self):
        """
        Mutation operation. Change the values of a gene.
        """
        self.pos_1 = (random.randint(0, 500), random.randint(0, 500))
        self.pos_2 = (random.randint(0, 500), random.randint(0, 500))
        self.pos_3 = (random.randint(0, 500), random.randint(0, 500))
        self.color = {'r': random.randint(0, 255),
                      'g': random.randint(0, 255),
                      'b': random.randint(0, 255),
                      'a': 128}


class genetic(object):
    """
    The genetic operation utils class.
    """

    def crossover(self, parent_1, parent_2, tri_num):
        """

        Transform the parent genes into array representation(DNA sequence).
        The child get a portion of DNA from parent_1, then from parent_2,
        then from parent_1 again, and so on.
        Args:
            parent_1: a parent.
            parent_2: another parent.
            tri_num: number of triangles that form the result image.
        Returns:
            new_genes: an array representation(DNA sequence) of the gene.
        """
        array1 = parent_1.get_array()
        array2 = parent_2.get_array()
        new_array = []
        flag = -1
        last_pos = 0
        pos = random.randint(0, 50)
        while last_pos < tri_num * 6:
            if flag > 0:
                new_array += array1[last_pos:pos]
            else:
                new_array += array2[last_pos:pos]
            flag *= -1
            last_pos = pos
            pos = random.randint(last_pos, last_pos+50)
        return new_array

    def mutate(self, genes, rate, tri_num):
        """
        Mutation Selector:
        Select some genes and let them mutate.
        Args:
            genes: the list of genes of an individual.
            rate: the mutation rate.
            tri_num: number of triangles that form the result image.
        Returns:
            genes: a list of mutated genes.
        """
        if random.uniform(0, 1) < rate:
            mut_genes = random.sample(list(range(tri_num)), 5)
            for g in mut_genes:
                genes[g].mutate()
        return genes


TARGET_IMG_NAME = 'Symmetra.jpg'  # target image file name
POP_SIZE = 25  # population size
MUT_RATE = 0.05  # mutation rate
GENERATIONS = 100000  # number of generations
CHILDREN_PER_GEN = 5  # children generated in each generations
TRI_NUM = 100  # number of triangles
BACKGROUND_COLOR = 'white'  # background color

# load target image and resize
TARGET = Image.open(TARGET_IMG_NAME).convert('RGB')
TARGET = TARGET.resize((256, 256))


class individual(object):
    """
    The individual class, which is an image.
    Attributes:
        genes: all the genes(triangles).
        im: the actual image file.
        fitness: the fitness, the lower the better
    """

    def __init__(self, parent_1=None, parent_2=None):
        """
        Inits an individual.
        If it has parents, generate its genes via crossover operation.
        If not, generate its genes randomly.
        Args:
            parent_1: a parent.
            parent_2: another parent.
        """
        self.genes = []
        op = genetic()
        if parent_1 and parent_2:
            array = op.crossover(parent_1, parent_2, TRI_NUM)
            self.generate_genes_from_array_with_mut(array, MUT_RATE)
        else:
            for i in range(TRI_NUM):
                self.genes.append(gene())
        # set actual image
        self.im = self.get_current_img()
        # calculate fitness
        self.fitness = self.get_fitness(TARGET)

    def get_current_img(self):
        # Drawing the genotype
        im = Image.new('RGB', (256, 256), BACKGROUND_COLOR)
        draw = ImageDraw.Draw(im, 'RGBA')
        for gene in self.genes:
            draw.polygon([gene.pos_1, gene.pos_2, gene.pos_3],
                         fill=(gene.color['r'], gene.color['g'],
                               gene.color['b'], gene.color['a']))
        del draw
        return im

    def save_current_img(self, f_name):
        # Saving an image of current generation
        self.im.save(f_name, 'PNG')

    def get_fitness(self, target):
        """

        This is what I was trying to come up with, difference between pixels and RMS:
        fitness = 0
        sourceHeight = target.size[0]
        sourceWidth = target.size[1]
        for y in range(sourceWidth):
            for x in range(sourceHeight):
                src = target.getpixel((x, y))
                gn = self.im.getpixel((x, y))
                sp1 = src[0]
                sp2 = src[1]
                sp3 = src[2]
                tp1 = gn[0]
                tp2 = gn[1]
                tp3 = gn[2]
                delta1 = sp1 - tp1
                delta2 = sp2 - tp2
                delta3 = sp3 - tp3

                pixelFitness = sqrt((square(delta1) + square(delta2) + square(delta3))/3)
                fitness = fitness + pixelFitness
        return fitness

        The below is the same idea, it is using the histogram difference to calculate fitness
        """
        h = ImageChops.difference(target, self.im).histogram()
        return math.sqrt(reduce(operator.add,
                                list(map(lambda h, i: h * (i ** 2),
                                         h, list(range(256)) * 3))) /
                         (float(target.size[0]) * target.size[1]))






    def get_array(self):
        # Array of genes
        array = []
        for g in self.genes:
            array.append(g.pos_1)
            array.append(g.pos_2)
            array.append(g.pos_3)
            array.append(g.color['r'])
            array.append(g.color['g'])
            array.append(g.color['b'])
        return array

    def generate_genes_from_array_with_mut(self, array, rate):
        # Mutating the genes
        new_array = list(zip(*[iter(array)] * 6))
        self.genes = []
        for chunk in new_array:
            g = gene()
            if random.uniform(0, 1) > rate:
                g.pos_1 = chunk[0]
                g.pos_2 = chunk[1]
                g.pos_3 = chunk[2]
                g.color['r'] = chunk[3]
                g.color['g'] = chunk[4]
                g.color['b'] = chunk[5]
            self.genes.append(g)


def initialize(pop):
    # Initialize population
    for i in range(POP_SIZE * 2):
        pop.append(individual())
    pop.sort(key=lambda x: x.fitness)
    pop = pop[:POP_SIZE]
    return pop


def evolve(pop):
    # Evolution function
    for i in range(GENERATIONS):

        children = []
        # generate weighed choices according to fitness
        parent_choices = []
        w = 100
        for p in pop:
            parent_choices.append((p, w))
            if w > 0:
                w = w - 10
        pop_choices = [val for val, cnt in parent_choices for j in range(cnt)]
        # generate children
        for j in range(CHILDREN_PER_GEN):
            parent_1 = random.choice(pop_choices)
            parent_2 = random.choice(pop_choices)
            child = individual(parent_1, parent_2)
            children.append(child)
        # compare and save new individuals
        pop += children
        pop.sort(key=lambda x: x.fitness)
        pop = pop[:POP_SIZE]
        # print log info
        if i % 10000 == 0 or i in [10, 100, 200, 500, 1000, 5000]:
            pop[0].save_current_img(str(i) + '_b.png')  # save intermediate imgs
        if i % 10 == 0:
            # print current best fitness and avg fitness
            avg = sum([x.fitness for x in pop]) / 25
            print("Finish " + str(i), pop[0].fitness, avg)
    return pop


if __name__ == '__main__':
    pop = []
    pop = initialize(pop)
    pop = evolve(pop)
    pop[0].save_current_img('best.png')
    print(pop[0].fitness)
