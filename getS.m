function s = getS(a,b)

for i=1:a
  for j=1:b
    s(i,j) = sqrt((5+2*i)*(5+2*j))/(5+i+j);
  end
end
