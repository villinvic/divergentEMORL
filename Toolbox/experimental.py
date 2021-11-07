
class Switch:
    def __init__(self, case_dict):
        self.case_dict = case_dict

    def __call__(self, input):
        if input not in self.case_dict:
            return None
        return self.case_dict[input]()


if __name__ == '__main__':
    switch = Switch({
        0 : lambda : print('LoL'),
        1 : lambda : print('ROFL'),
    })

    switch(0)
    switch(1)
    switch(2)