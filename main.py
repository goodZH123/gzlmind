import torch    
def main():
    print("Hello, gzlMind!")
    # mask = torch.tensor([[False, True, False], [True, False, False]])
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = x.new_zeros(1).squeeze()
    print(y)

    
if __name__ == "__main__":
    main()
