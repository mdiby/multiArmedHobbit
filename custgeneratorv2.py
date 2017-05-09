import numpy as np
import random
from collections import namedtuple
from cachetools import LRUCache


class customerMaker():
    def __init__(self):
        # establish categories
        Species = namedtuple('Species', 'orc troll hobbit human dwarf elf wizard dragon')
        States = namedtuple('States', 'species magic_level power_hunger status')
        self.species = Species((40,2,2,2,2,40), (10,8,7,1,2,20), (10,8,7,1,2,1), (8,11,10,1,2,1),
                               (8,7,1,1,8,10), (2,1,2,5,5,3), (1,3,3,8,95,1), (1,3,5,7,8,110))
        # self.species = Species((400,20,20,20,20,400),(100,800,70,10,20,200),(100,80,70,10,20,10),(80,110,100,10,20,10),
        #                       (80,70,10,10,80,100),(20,10,20,50,50,30),(10,30,30,80,950,10),(10,30,50,70,80,1100))
        self.magic_levels = set(['beginner', 'medium', 'advanced'])
        self.power_hunger = set(['unambitious', 'thirsty'])
        self.status = set(['proletariat', 'bourgeoisie', 'landlord'])
        self.prices = range(6)
        self.context = States(None, None, None, None)
        self.wtp = None

    def returnCoins(self, w):
        '''Given a vector of rewards and respective probablity of occurance, will return reward'''
        w = np.array(w)
        r = random.uniform(0, sum(w))

        # if the random variable is less than our value, return the corresponding entity
        for n, v in zip(self.prices, [sum(w[:x+1]) for x in range(len(w))]):
            if r < v:
                return n

    def getContext(self):
        '''Randomly selects from each feature to return a context vector'''
        # WORK ON MAKING THESE CORRELATED IN THE FUTURE!
        # pick a species
        self.context = self.context._replace(species=random.choice(list(self.species._asdict().keys())))
        # pick a magic level
        self.context = self.context._replace(magic_level=random.choice(list(self.magic_levels)))
        # pick a power_hunger level
        self.context = self.context._replace(power_hunger=random.choice(list(self.power_hunger)))
        # pick a status level
        self.context = self.context._replace(status=random.choice(list(self.status)))
        # reset wtp
        self.wtp = None
        return self.context.species, self.context.magic_level, self.context.power_hunger, self.context.status

    # for tensorflow, we want to get the numeric context to feed into one-hot
    def contextToNum(self):
        '''Randomly selects from each feature to return a context vector'''
        species_num = list(self.species._asdict().keys()).index(self.context.species)
        magic_num = list(self.magic_levels).index(self.context.magic_level)
        power_num = list(self.power_hunger).index(self.context.power_hunger)
        status_num = list(self.status).index(self.context.status)
        return species_num, magic_num, power_num, status_num

    def getWtp(self, sp_index=None, p_wtp=None, p_rewards=None):
        '''Given a context vector, reveals the reward for the action as well 'true' reward'''
        # get the dirichlet parameters for the species
        if sp_index is None:  # if this is a new customer, draw from the distribution
            species = self.context.species
            params = self.species._asdict()[species]
            # sample from the distribution once
            probs = np.random.dirichlet(params,1)

            # additional logic to add non-linear complexity to the data
            if self.context.status is 'landlord':
                # they can afford a more expensive price
                probs[0][4] *= 1.5
                probs[0][5] *= 2

            elif self.context.status is 'bourgeoisie':
                # can afford the second most expensive price
                probs[0][3] *= 1.5
                probs[0][4] *= 2

            else:
                # can't afford stuff (squashes the probability)
                probs[0][0] *= 4
                probs[0][1] *= 3

            if self.context.power_hunger is 'thirsty':
                # wants ring more
                if species is 'hobbit': #Bilbo
                    # This guys really really needs the ring
                    probs[0][5] *= 5
                elif species in ('dragon','elf'):
                    probs[0][4] *= 1.2
                    probs[0][5] *= 5
                elif species is 'human':
                    probs[0][3] *= 2
                else:
                    pass
            else:
                if species is 'hobbit':
                    probs[0][0] *= 3
                elif species is 'dwarf':
                    probs[0][2] *= 1.5
                else:
                    probs[0][1] *= 1.25

            if self.context.magic_level is 'advanced':
                # good magicians know your rings aren't actually magic!
                probs[0][0] *= 2

            elif self.context.magic_level is 'medium':
            # even medium-magic elves know your ring is magically worthless unless they like pretty things in which case
            # they'll buy your ring anyway
                if species is 'elf' and self.context.status is not 'landlord':
                    probs[0][0] *= 2
                elif self.context.power_hunger is 'unambitious':
                    probs[0][1] *= 2
                else:
                    probs[0][1] *= 1.2

            else:
                probs[0][3] *= 1.5


            # rescale your probablities to fall from 0-1
            probs = probs[0]/np.sum(probs[0])
            # get the willingness to pay
            self.wtp = self.returnCoins(probs)

        else:  # if this is an old customer
            # get species:
            species = list(self.species._asdict().keys())[sp_index]
            # if they had a deal, increase willingness to pay by 1 until you max out
            if species in ('hobbit', 'human', 'dwarf'):
                if p_rewards[-1] is not 0:
                    self.wtp = p_wtp + 1 if p_wtp < 5 else 5
                else:  # if there was no deal, decrease wtp by 1 until hit 0
                    self.wtp = p_wtp - 1 if p_wtp > 0 else 0
            elif species in ('elf', 'wizard', 'dragon'):
                # these creatures have long-term memory so they only care about whether the first deal was successful
                if p_rewards[0] is not 0:
                    self.wtp = p_wtp + 2 if p_wtp < 4 else 5
                else:  # if there was no deal, decrease wtp by 1 until hit 0
                    self.wtp = p_wtp - 2 if p_wtp > 1 else 0
            else:
                #for orcs and trolls, they're stupid so they have no memory
                self.wtp = p_wtp

        return self.wtp

    def acceptOffer(self, action):
        if self.wtp >= action:
            return 1
        else:
            return 0


class customerFeeder():
    '''create world where customers can buy up to maxsize purchases'''
    def __init__(self, maxsize=5):
        # establish categories
        self.maxsize = maxsize
        caches = namedtuple('caches', 'model zero one two three four five')
        cycles = namedtuple('cycles', 'model zero one two three four five')
        self.cache = caches(LRUCache(maxsize=self.maxsize), LRUCache(maxsize=self.maxsize),
                            LRUCache(maxsize=self.maxsize), LRUCache(maxsize=self.maxsize),
                            LRUCache(maxsize=self.maxsize), LRUCache(maxsize=self.maxsize),
                            LRUCache(maxsize=self.maxsize))
        self.cycle = cycles(0, 0, 0, 0, 0, 0, 0)
        self.cust = customerMaker()
        self.newbies = None
        # for storage purposes
        self.context = None
        self.wtp = None
        self.cust_id = None  # keep track of current customer identity
        self.history = None

    def getNewCust(self):
        'reveals context of new customer'
        # customer iterator MUST use i
        self.cust_id = 0 if len(self.cache.model.keys()) is 0 else (max(self.cache.model.keys()) + 1)  # get last cust id
        self.cust.getContext()  # reveal state
        self.context = self.cust.contextToNum()
        self.wtp = self.cust.getWtp()  # get max price

    def getOldCust(self, cachename='model'):
        'reveals context of new customer'
        cache = getattr(self.cache, cachename)
        cycle = getattr(self.cycle, cachename)
        self.cust_id = min(cache.keys()) + cycle
        return_cust = cache[self.cust_id]
        self.context = return_cust['context']
        # input the last wtp and last deal to get new wtp
        self.wtp = self.cust.getWtp(self.context[0], return_cust['wtp'][-1], return_cust['reward']) #get max price
        self.history = return_cust['reward']


    def feedNewCust(self, action, cachename='model'):
        'Method that takes place when a new customer gets cached'
        deal = self.cust.acceptOffer(action) #reveal deal status
        # save to the cache
        cache = getattr(self.cache, cachename)
        cache[self.cust_id] = {'context':self.context, 'wtp':[self.wtp], 'reward':[deal * action]} #save to our cache
        self.cycle = self.cycle._replace(**{cachename: 0})

        return deal


    def feedOldCust(self, action, cachename='model'):
        'Method that takes place when a return customer gets cached'
        # find the last return customer who hasn't made a purchase
        cache = getattr(self.cache, cachename)
        cycle = getattr(self.cycle, cachename)
        return_cust = cache[self.cust_id]
        deal = self.cust.acceptOffer(action)
        # update customer history
        return_cust['wtp'].append(self.wtp),
        return_cust['reward'].append(deal * action)
        # once the customer makes 5 purchases, dump them from cache
        # increment your cycle to reference the next returned customer
        self.cycle = self.cycle._replace(**{cachename: cycle + 1})

        if len(return_cust['wtp']) is self.maxsize:  # delete customer if they reach maxsize
            cache.pop(self.cust_id)
            # reset the cycle if you delete them
            self.cycle = self.cycle._replace(**{cachename: 0})

        self.context, self.wtp, self.cust_id = None, None, None  # reset

        return deal

    def forecastNewCust(self, iters=100):
        'Calculates which cache indices have to insert new customers in the stream'
        x = [0]
        max_num = max(x)
        i = 1
        while iters > max_num:
            x.append(x[-1] + i)
            max_num = x[-1]
            i += 1
            i = i if i < self.maxsize else self.maxsize
        self.newbies = x
