from util.controls import *
from warnings import filterwarnings
from models.neural_net import *
from sklearn.pipeline import Pipeline

filterwarnings("ignore")

if __name__ == "__main__":

    X_train, X_test, y_train, y_test = read_bin("temp_data/271_all.pkl")
    n_feat = X_train.shape[1]
    x_pipeline = Pipeline(
        [
            ('scaler', MinMaxScaler()),
        ],
        verbose=1,
    )

    X_train = x_pipeline.fit_transform(X_train)
    X_test = x_pipeline.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    X_train, X_test, y_train = torch.Tensor(X_train), torch.Tensor(X_test), torch.Tensor(y_train)
    X_train, X_test, y_train = X_train.cuda(), X_test.cuda(), y_train.cuda()
    
    model = LinearNet(n_feat=n_feat)
    model = model.to(device)

    epochs = 1000
    optimizer = optim.Adam
    loss_func = nn.MSELoss()
    learning_rate = 1e-4
    batch_size = None

    model, history = train(
        model,
        X_train,
        y_train,
        X_test, 
        y_test,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        optimizer=optimizer,
        loss_func=loss_func,
    )

    y_pred = model(X_test).detach().cpu().numpy().reshape(-1,)
    y_pred_train = model(X_train).detach().cpu().numpy().reshape(-1,)
    y_train = y_train.detach().cpu().numpy().reshape(-1,)

    test_corr = correlation(y_pred, y_test)
    train_corr = correlation(y_pred_train, y_train)
    print("Test Correlation: ", round(test_corr, 8))
    print("Train Correlation: ", round(train_corr, 8))
    plt.plot(list(range(len(history))), history)
    plt.show()
    log_model(
        model,
        corr=[test_corr, train_corr],
        learning_rate=learning_rate,
        epochs=epochs,
        loss=str(loss_func)[:-2],
        optimizer=optimizer.__name__,
        notes='With Feature Pipeline (min max scaler)'
    )
