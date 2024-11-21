def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup data loaders
    train_loader, test_loader = setup_data_loaders(args)
    
    # Initialize model
    model = setup_model(args, device)
    if args.arch_type == "urdmu":
        model = VideoPruningWrapper(model)
    
    # Initialize trainer
    criterion = nn.CrossEntropyLoss()
    trainer = LotteryTicketTrainer(args, model, train_loader, test_loader, criterion, device)
    
    # Train with pruning
    trainer.initialize_training()
    compression, accuracy = trainer.train_with_pruning()
    
    # Plot results
    plot_results(compression, accuracy, args)

if __name__ == "__main__":
    main()