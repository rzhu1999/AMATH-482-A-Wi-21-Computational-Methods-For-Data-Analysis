function [U,S,V,threshold,w,sort_first,sort_second] = digit_trainer(digitData1, digitData2, feature)
    
    feature = 20;
    n_first = size(digitData1,2);
    n_second = size(digitData2,2);
    [U, S, V] = svd([digitData1 digitData2], 'econ');
    digits = S*V'; % projection onto principal components: X = USV' --> U'X = SV'
    first = digits(1:feature,1:n_first);
    second = digits(1:feature,n_second+1:n_first+n_second);
    m_first = mean(digitData1,2);
    m_second = mean(digitData2,2);

    Sw = 0; % within class variances
    for k=1:n_first
        Sw = Sw + (digitData1(:,k) - m_first)*(digitData2(:,k) - m_first)';
    end

    for k=1:n_second
        Sw = Sw + (digitData2(:,k) - m_second)*(digitData2(:,k) - m_second)';
    end

    Sb = (m_first-m_second)*(m_first-m_second)'; % between class

    [V2, D] = eig(Sb,Sw); % linear disciminant analysis
    [lambda, ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);

    v_first = w'*digitData1;
    v_second = w'*digitData2;

    if mean(v_first) > mean(v_second) %Make first digit always below
        w = -w;
        v_first = -v_first;
        v_second = -v_second;
    end

    sort_first = sort(v_first);
    sort_second = sort(v_second);
    t1 = length(sort_first);
    t2=1;
    
    while sort_first(t1)>sort_second(t2)
        t1 = t1-1;
        t2 = t2+1;
    end

    threshold = (sort_first(t1)+sort_second(t2))/2;

end