import pgl
import paddle

def load(name):
    if name == 'cora':
        dataset = pgl.dataset.CoraDataset(self_loop=True)
    elif name == "pubmed":
        dataset = pgl.dataset.CitationDataset("pubmed", self_loop=True)
    elif name == "citeseer":
        dataset = pgl.dataset.CitationDataset("citeseer", self_loop=True)
    else:
        raise ValueError(name + " dataset doesn't exists")
    return dataset

def load_data(name):
    dataset = load(name)
    graph = dataset.graph.tensor()
    features = paddle.to_tensor(dataset.graph.node_feat['words'])
    labels = paddle.to_tensor(dataset.y)
    num_classes = paddle.to_tensor(dataset.num_classes)
    train_index = paddle.to_tensor(dataset.train_index)
    val_index = paddle.to_tensor(dataset.val_index)
    test_index = paddle.to_tensor(dataset.test_index)
    return graph, features, labels, num_classes, train_index, val_index, test_index

def evaluate(logits, labels, index):
    logits = paddle.gather(logits, index)
    labels = paddle.gather(labels, index)
    indices = paddle.argmax(logits, axis=1)
    correct = paddle.sum(paddle.cast(indices == labels, dtype='int'))
    return correct.numpy() * 1.0 / len(labels)