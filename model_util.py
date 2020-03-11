#-*- coding: utf-8 -*-


class EarlyStopping:
    
    def __init__(self, patience=5, for_descending=False):
        self.patience       = patience
        self.for_descending = for_descending
        self.counter        = 0
        if self.for_descending:
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")
            
    def add_value(self, value):
        
        is_improved = False
        
        if self.for_descending:
            if value < self.best_value:
                self.best_value = value
                self.counter = 0
                is_improved = True
            
        else:
            if value > self.best_value:
                self.best_value = value
                self.counter = 0
                is_improved = True
                
        if ~is_improved:
            self.counter += 1
            
        return is_improved
            
    
    def should_stop(self):
        return self.counter >= self.patience

