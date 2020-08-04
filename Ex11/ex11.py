from surprise import NMF
from surprise.model_selection import train_test_split
from surprise import Dataset
from surprise import Reader

reader = Reader(line_format='user item rating', sep=',' ,rating_scale=(0,10) )
data = Dataset.load_from_file('user_ratings1.data', reader=reader)
data.raw_ratings

trainset, testset = train_test_split(data, test_size = .7, train_size = .3, random_state = 1)

algo = NMF(n_factors=25, n_epochs=200, random_state=1)
#trainset = data.build_full_trainset()
algo.fit(trainset)


from surprise import accuracy
pred = algo.test(testset)

accuracy.rmse(pred), accuracy.mae(pred)


for i in range(20):
    print('user: %-10s      item: %-10s    r_ui: %-10s    est: %.2f     %-10s' % (pred[i].uid, pred[i].iid, pred[i].r_ui , pred[i].est , pred[i][-1]))
    
