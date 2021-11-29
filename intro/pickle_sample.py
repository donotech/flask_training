import pickle


class AddressBook:
    def __init__(self, name, phone, address):
        self.name = name
        self.phone = phone
        self.address = address

    def __str__(self) -> str:
        return super().__str__()

    def print_self(self):
        print("Na = " + self.name + " ph " + str(self.phone) + " addr " + self.address)


def main_dump():
    a1 = AddressBook("a", 1234, "addressA")
    fp = open("address_book.pkl", "wb")
    pickle.dump(a1, fp)


def main_load():
    fp = open("address_book.pkl", "rb")
    pkl_address = pickle.load(fp)
    pkl_address.print_self()


if __name__ == "__main__":
    #main_dump()
    main_load()



