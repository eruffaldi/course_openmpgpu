
struct Node
{
	int value;
	std::vector<Node*> children;
};

int counter = 0;

void process(Node *p)
{
	#pragma omp critical pippo
	counter += p->value;
}

void visit_dfs_pre_rec(Node *p)
{
	if(!p)
		return;
	process(p);
	for(auto & c : p->children)
		visit_dfs_pre_rec(c);
}


void visit_dfs_post_rec(Node *p)
{
	if(!p)
		return;
	for(auto & c : p->children)
		visit_dfs_pre_rec(c);
	process(p);
}

void visit_dfs_pre_nonrec(Node *p)
{
	if(!p)
		return;
	std::stack<Node*> todo;
	todo.push(p);
	while(!todo.empty())
	{
		Node * q = todo.top();
		todo.pop();
		process(q);
		for(auto & c: q->children)
			todo.push(c);	
	}
}

void visit_dfs_post_nonrec(Node *p)
{
	if(!p)
		return;
	struct State
	{
		Node * p;
		bool firstpass = true;
	};
	std::stack<State> todo;
	todo.push({p,true});
	while(!todo.empty())
	{
		State s = todo.top();
		todo.pop();
		if(s.firstpass)
		{
			for(auto & c: s.p->children)
				todo.push({c,true});	
			todo.push({p,false});
		}
		else
		{
			process(s.p);
		}
	}
}


void visit_dfs_pre_rec_omp(Node *p)
{
	if(!p)
		return;
	process(p);
	for(auto & c : p->children)
		visit_dfs_pre_rec(c);
}

void visit_any_rec_omp(Node *p)
{
	if(!p)
		return;
	for(auto & c : p->children)
	{
		#pragma omp task 
		visit_dfs_pre_rec(c);
	}
	#pragma omp taskwait
	process(p);
}

int myvisit(Node * p)
{
	#pragma omp parallel
	{
		#pragma omp single
		{
			visit_any_rec_omp(p);
		}
	}
}