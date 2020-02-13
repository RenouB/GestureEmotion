import torch
from torch.nn import functional as F
from torch import nn
from layers import CNN
torch.manual_seed(200)

class OneActorOneModalityBrute(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, num_classes, dropout):

		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.classify = nn.Linear(project_output_dim, num_classes)

		print(self)

	def forward(self, feats):
		feats = self.dropout(feats)

		feats = self.cnn_block(feats)
		feats = self.project(feats)
		feats = self.classify(feats)
		out = torch.sigmoid(feats)
		return out, torch.Tensor([[0,0,0]])

class OneActorOneModalityDeltas(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, attention_dim,
				 num_classes, dropout):

		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.cnn_block1 = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.cnn_block2 = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
								nn.ReLU())
		self.cnn_block3 = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
						nn.ReLU())

		self.project1 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		self.project2 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		self.project3 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		self.evidence1 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		self.evidence2 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		self.evidence3 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		self.attention_vector = nn.Sequential(nn.Linear(attention_dim, 1), nn.ReLU(), nn.Dropout())
		self.softmax = nn.Softmax(dim=1)
		self.classify = nn.Linear(project_output_dim, num_classes)

	def forward(self, poses, deltas, delta_deltas):
		poses, deltas, delta_deltas = [self.dropout(input) for input in
										[poses, deltas, delta_deltas]]

		poses = self.cnn_block1(poses)
		poses = self.project1(poses)
		evidence1 = self.evidence1(poses)
		score1 = self.attention_vector(evidence1)

		deltas = self.cnn_block2(deltas)
		deltas = self.project2(deltas)
		evidence2 = self.evidence2(deltas)
		score2 = self.attention_vector(evidence2)

		delta_deltas = self.cnn_block3(delta_deltas)
		delta_deltas = self.project3(delta_deltas)
		evidence3 = self.evidence2(delta_deltas)
		score3 = self.attention_vector(evidence3)
		scores = self.softmax(torch.cat([score1, score2, score3],dim=1))
		context = scores[:,0,None]*poses + scores[:,1,None]*deltas + scores[:,2,None]*delta_deltas
		context = self.classify(context)
		out = torch.sigmoid(context)
		return out, scores.detach()

class TwoActorsOneModalityCNN(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, att_vector_dim,
				 num_classes, dropout):

		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.attend = Attention(project_output_dim, att_vector_dim, dropout)
		self.classify = nn.Linear(project_output_dim, num_classes)

	def forward(self, feats1, feats2):

		feats1 = self.dropout(feats1)
		feats1 = self.cnn_block(feats1)
		feats1 = self.project(feats1)

		feats2 = self.dropout(feats2)
		feats2 = self.cnn_block(feats2)
		feats2 = self.project(feats2)

		attend1 = self.attend(feats1, feats2)
		attend1 = self.dropout(attend1)
		attend2 = self.attend(feats2, feats1)
		attend2 = self.dropout(attend2)

		out1 = self.classify(attend1)
		out1 = torch.sigmoid(out1)

		out2 = self.classify(attend2)
		out2 = torch.sigmoid(out2)

		out = torch.cat([out1, out2], dim=1).reshape((-1,1))
		return out

class TwoActorsOneModalitySimpleCNN(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, att_vector_dim,
				 num_classes, dropout):

		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.attend = SimpleAttention(project_output_dim, att_vector_dim, dropout)
		self.classify = nn.Linear(project_output_dim, num_classes)

	def forward(self, feats1, feats2):

		feats1 = self.dropout(feats1)
		feats1 = self.cnn_block(feats1)
		feats1 = self.project(feats1)

		feats2 = self.dropout(feats2)
		feats2 = self.cnn_block(feats2)
		feats2 = self.project(feats2)

		attend1 = self.attend(feats1, feats2)
		attend1 = self.dropout(attend1)
		attend2 = self.attend(feats2, feats1)
		attend2 = self.dropout(attend2)

		out1 = self.classify(attend1)
		out1 = torch.sigmoid(out1)

		out2 = self.classify(attend2)
		out2 = torch.sigmoid(out2)

		out = torch.cat([out1, out2], dim=1).reshape((-1,1))
		return out

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
