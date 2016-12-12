% matrix 2d iterators
syms stride real;
syms x y X Y W H real;
addressof_full = @(x,y,stride1,stride2) sum([x,y].*[stride1,stride2]);
addressof = @(x,y) addressof_full(x,y,1,stride);

X=0;
Y=0;

% by row
byrow = [];
byrow.inc1 = 1;
byrow.mod1 = W;
byrow.inc2 = addressof(X,y+1)-addressof(X+W-1,y);
simplify(byrow.inc2)

% by col
bycol = [];
bycol.inc1 = stride;
bycol.mod1 = H;
bycol.inc2 = addressof(x+1,Y)-addressof(x,Y+H-1);
bycol.inc2

% by row rev
byrowrev = [];
byrowrev.inc1 = -1;
byrowrev.mod1 = W;
byrowrev.inc2 = addressof(X+W-1,y-1)-addressof(X,y);
simplify(byrowrev.inc2)

% by col rev
bycolrev = [];
bycolrev.inc1 = -stride;
bycolrev.mod1 = H;
bycolrev.inc2 = addressof(x-1,Y+H-1)-addressof(x,Y);
bycolrev.inc2
