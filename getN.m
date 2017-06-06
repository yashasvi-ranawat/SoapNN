function s = getN(a,rcut)

for i=1:a
    s(i) = sqrt(rcut**(2*i + 5)/(2*i + 5));
end
