import sympy

import pandas as pd
from genetic_algo import *



class GenBalance:

    def __init__(self, df, attr, class_name, weighted_indexes, integer_indexes, random_indexes, regular_indexes, model):
        self.df = df
        self.attr = attr
        self.class_name = class_name
        self.weighted_indexes = weighted_indexes
        self.integer_indexes = integer_indexes
        self.random_indexes = random_indexes
        self.regular_indexes = regular_indexes
        self.model = model
        self.attributes = [col for col in self.df.columns if col != self.class_name]
        
    def fit(self):
        from sympy.abc import x
        for att in self.attributes:
            if att not in [self.attr]:
                self.regular_indexes.append((get_index(att, self.attributes)))

        X = self.df[self.attributes].values
        y = self.df[self.class_name].to_list()
        self.model.fit(X, y)

        values = self.df[self.attr].unique()
        val = values[0]
        other = values[1]

        aPP = self.df[(self.df[self.attr] != val) & (self.df[self.class_name] == 1)].values.tolist()
        aPN = self.df[(self.df[self.attr] != val) & (self.df[self.class_name] == 0)].values.tolist()
        aDP = self.df[(self.df[self.attr] == val) & (self.df[self.class_name] == 1)].values.tolist()
        aDN = self.df[(self.df[self.attr] == val) & (self.df[self.class_name] == 0)].values.tolist()

        discrimination = len(aPP) / (len(aPP) + len(aPN)) - len(aDP) / (len(aDP) + len(aDN))
        print("Disc:", discrimination)

        
        if discrimination == 0:
            print("No discrimination")
            return self.df
        elif discrimination > 0:
            self.PP = aPP
            self.PN = aPN
            self.DP = aDP
            self.DN = aDN

            print(val, "is discriminated")

            self.d = val
            self.p = other
        elif discrimination < 0:
            print(other, "is discriminated")

            self.PP = aDP
            self.PN = aDN
            self.DP = aPP
            self.DN = aPN

            self.p = val
            self.d = other
            
        self.budget = len(self.PP + self.DP) - len(self.DN + self.PN)
        self.budget = round(abs(self.budget))

        self.lPP = len(self.PP)
        self.lPN = len(self.PN)
        self.lDP = len(self.DP)
        self.lDN = len(self.DN)

        self.modPP = 0
        self.modPN = 0
        self.modDP = 0
        self.modDN = 0

        if (self.lPP + self.lDP) > (self.lDN + self.lPN):
            x = int(round(sympy.solve(((self.lPP)/(self.lPP+self.lPN + x)) 
                                        - ((self.lDP)/(self.lDP+self.lDN + self.budget - x)))[0]))
            print("- Budget aka Negative Records to Add:", self.budget, len(self.df)+self.budget,
                  "\n- PN to add:", x, 
                  "\n- DN to add:", self.budget-x, 
                  "\n- Disc:", ((self.lPP)/(self.lPP+self.lPN + x)) - ((self.lDP)/(self.lDP+self.lDN + self.budget - x)))
            self.modPN = x
            self.modDN = self.budget-x

        elif (self.lPP + self.lDP) < (self.lDN + self.lPN):
            print("DP: Always added")
            print("PP: Maybe removed")
            x = int(round(sympy.solve(((self.lPP+self.budget - x)/(self.lPP+self.lPN + self.budget - x)) -
                                        ((self.lDP + x)/(self.lDP+self.lDN + x)))[0]))
            print("- Budget aka Positive Record to Add:", self.budget, len(self.df)+self.budget,
                  "\n- PP to add:", self.budget - x,
                  "\n- DP to add:", x,
                  "\n- Disc:", ((self.lPP+self.budget - x)/(self.lPP+self.lPN + self.budget - x)) 
                  - ((self.lDP + x)/(self.lDP+self.lDN + x)))
            self.modDP = x
            self.modPP = self.budget-x
            
            
        if self.modDP > 0: #HIGH PROBA as fitness
            const = [(get_index(self.attr, self.attributes), self.d)]
            num = self.modDP
            fit = True
            new_records = GA(self.DP, const, num, self.model, fit, self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes)
            for rec in new_records:
                record = rec[0]
                record.append(1)
                self.DP.append(record)

        if self.modPP > 0: #LOW PROBA as fitness
            const = [(get_index(self.attr, self.attributes), self.p)]
            num = self.modPP
            fit = False
            new_records = GA(self.PP, const, num, self.model, fit, self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes)
            for rec in new_records:
                record = rec[0]
                record.append(1)
                self.PP.append(record)

        if self.modDN > 0: #LOW PROBA as fitness
            const = [(get_index(self.attr, self.attributes), self.d)]
            num = self.modDN
            fit = False
            new_records = GA(self.DN, const, num, self.model, fit, self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes)
            for rec in new_records:
                record = rec[0]
                record.append(0)
                self.DN.append(record)

        if self.modPN > 0: #HIGH PROBA as fitness
            const = [(get_index(self.attr, self.attributes), self.p)]
            num = self.modPN
            fit = True
            new_records = GA(self.PN, const, num, self.model, fit, self.weighted_indexes, self.integer_indexes, self.random_indexes, self.regular_indexes)
            for rec in new_records:
                record = rec[0]
                record.append(0)
                self.PN.append(record)
            
        
        return pd.DataFrame.from_records(self.DP + self.PP + self.DN + self.PN, columns=self.attributes + [self.class_name])