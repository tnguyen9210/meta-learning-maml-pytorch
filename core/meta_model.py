import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.meta_net import MetaConvNet

class MetaModel(object):
    def __init__(self, args):
        self.device = args['device']
        self.num_iters = args['num_iters']
        self.num_episodes = args['num_episodes']
        
        # neuralnet and losses
        self.meta_net = MetaConvNet(args)
        self.cls_crit = nn.CrossEntropyLoss()

        self.meta_net.to(self.device)
        self.cls_crit.to(self.device)
        
        # optimizer and lr scheduler
        # self.optim = optim.Adam(self.net.parameters(), lr=args['lr'])
        self.meta_optim = optim.SGD(
            self.meta_net.parameters(), lr=args['lr'], momentum=args['momentum'],
            weight_decay=args['l2_params'])
        
    def update(self, train_lbl, epoch, logger):
        # switch net to train mode
        self.net.train()

        # make batch generators
        num_iters = len(train_lbl)
        interval = int(num_episodes/4) + 1
        train_lbl_iter = iter(train_lbl)
        print(num_iters)
        print(self.num_iters)
        print(self.num_episodes)
        stop
        
        # training
        train_loss = 0.
        start_time = time.time()
        
        outer_loss = torch.tensor(0., device=self.device)
        for it in range(self.num_iters):
            # get train lbl data
            support_x, support_y, query_x, query_y = train_lbl_iter.next()
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            for eps in range(self.num_episodes):
                # print(support_x.shape)
                # print(support_y.shape)
                # print(query_x.shape)
                # print(query_y.shape)

                # 2. inner loss
                # feed data and compute loss
                support_logit = self.meta_net(support_x)
                inner_loss = self.cls_crit(support_logit, support_y)

                # backprop and update 
                self.meta_net.zero_grad()
                params = gradient_update_params(self.meta_net, inner_loss)

                query_logit = self.meta_net(query_set)
                outer_loss += self.cls_crit(query_logit, query_y)

            outer_loss = outer_loss/num_episodes
            outer_loss.backward()
            meta_optim.step()
        
    def evaluate(self, eval_lbl):
        # switch net to eval mode
        self.net.eval()
        
        # evaluating
        num_eval = len(eval_lbl.dataset)
        eval_y_corrects = 0.
        with torch.no_grad():
            for i, batch in enumerate(eval_lbl):
                eval_x, eval_y = batch
                eval_x = eval_x.to(self.device)
                eval_y = eval_y.to(self.device)

                eval_logit = self.net(eval_x)
                
                eval_pred = torch.argmax(eval_logit, dim=1)
                eval_y_corrects += torch.sum(eval_pred == eval_y).item()

        acc = eval_y_corrects/num_eval*100
        return {'acc': round(acc, 4), 'error': round(100-acc, 4)}

    
    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cuda:0')
        self.net.load_state_dict(state_dict)
        
    def save_state(self, save_dir, epoch=None):
        if epoch:
            save_file = f"{save_dir}/epoch_{epoch}.ckpt"
        else:
            save_file = f"{save_dir}/best_model.ckpt"
        # prevent disruption during saving
        try:
            torch.save(self.net.state_dict(), save_file)
            print("model saved to {}".format(save_file))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

            
def gradient_update_params(
        model, loss, params=None, step_size=0.5, first_order=False):
    """ Update of the meta-parameters with one step of gradient descent on the
    loss function.
    """
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss,
                                params.values(),
                                create_graph=not first_order)

    updated_params = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size[name] * grad

    else:
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

    return updated_params
    
