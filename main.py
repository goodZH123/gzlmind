import torch    
def main():
    print("Hello, gzlMind!")
    x = torch.tensor([[1.0,0,0], [0,2.0,0], [0,0,0], [3.0,0,0]])
    print(x.shape)
    x = x.any(dim=-1)
    print(x)
    y = x.nonzero()
    print(y)
    y = y.flatten()
    print(y)

if __name__ == "__main__":
    main()
