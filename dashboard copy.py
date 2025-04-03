import React, { useState, useEffect } from 'react';
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import Papa from 'papaparse';
import _ from 'lodash';

const FinancialDashboard = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [companies, setCompanies] = useState([]);
  const [dateRange, setDateRange] = useState({ start: null, end: null });
  const [selectedQuarters, setSelectedQuarters] = useState({ quarter1: null, quarter2: null });
  const [quarters, setQuarters] = useState([]);

  // Colors for charts
  const COLORS = ['#3498DB', '#2ECC71', '#E74C3C', '#F1C40F', '#9B59B6', '#1ABC9C', '#D35400'];

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const response = await window.fs.readFile('paste.txt', { encoding: 'utf8' });
        
        // Extract the CSV data and parse it
        Papa.parse(response, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            // Process data
            const parsedData = results.data;
            
            // Format dates
            parsedData.forEach(row => {
              if (row['Quarter End Date']) {
                row['Quarter End Date'] = new Date(row['Quarter End Date']);
              }
            });
            
            setData(parsedData);
            
            // Extract unique companies
            const uniqueCompanies = [...new Set(parsedData.map(item => item.Company))].filter(Boolean);
            setCompanies(uniqueCompanies);
            
            // Set default selected company
            if (uniqueCompanies.length > 0) {
              setSelectedCompany(uniqueCompanies[0]);
            }
            
            setLoading(false);
          },
          error: (error) => {
            console.error("Error parsing CSV:", error);
            setError("Failed to parse data. Please check the format.");
            setLoading(false);
          }
        });
      } catch (error) {
        console.error("Error loading data:", error);
        setError("Failed to load data. Please try again.");
        setLoading(false);
      }
    };

    loadData();
  }, []);

  // Update quarters when company changes
  useEffect(() => {
    if (selectedCompany && data.length > 0) {
      const companyData = data.filter(item => item.Company === selectedCompany);
      
      // Get unique quarters for the selected company
      const uniqueQuarters = _.uniqBy(companyData, 'Quarter End Date')
        .map(item => item['Quarter End Date'])
        .filter(date => date instanceof Date && !isNaN(date))
        .sort((a, b) => a - b);
      
      setQuarters(uniqueQuarters);
      
      // Set date range based on available quarters
      if (uniqueQuarters.length > 0) {
        setDateRange({
          start: uniqueQuarters[0],
          end: uniqueQuarters[uniqueQuarters.length - 1]
        });
      }
    }
  }, [selectedCompany, data]);

  // Filter data based on selections
  const getFilteredData = () => {
    if (!selectedCompany) return [];
    
    let filtered = data.filter(item => item.Company === selectedCompany);
    
    if (dateRange.start && dateRange.end) {
      filtered = filtered.filter(item => 
        item['Quarter End Date'] >= dateRange.start && 
        item['Quarter End Date'] <= dateRange.end
      );
    }
    
    return filtered.sort((a, b) => a['Quarter End Date'] - b['Quarter End Date']);
  };

  // Get data for a specific quarter
  const getQuarterData = (quarterDate) => {
    if (!quarterDate) return null;
    
    return data.find(item => 
      item.Company === selectedCompany && 
      item['Quarter End Date'] && 
      item['Quarter End Date'].getTime() === quarterDate.getTime()
    );
  };

  // Format date for display
  const formatQuarterDate = (date) => {
    if (!date || !(date instanceof Date) || isNaN(date)) return 'N/A';
    
    const quarter = Math.floor((date.getMonth() / 3) + 1);
    return `${date.getFullYear()} Q${quarter}`;
  };

  // Format numbers with commas and 2 decimal places
  const formatCurrency = (value) => {
    if (value === undefined || value === null) return 'N/A';
    return `Rs. ${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  // Calculate percentage change
  const calculateChange = (current, previous) => {
    if (current === undefined || previous === undefined || previous === 0) return null;
    return ((current - previous) / Math.abs(previous)) * 100;
  };

  // Get latest financials
  const getLatestFinancials = () => {
    const filteredData = getFilteredData();
    if (filteredData.length === 0) return null;
    
    return filteredData[filteredData.length - 1];
  };

  // Get previous quarter financials
  const getPreviousFinancials = () => {
    const filteredData = getFilteredData();
    if (filteredData.length < 2) return null;
    
    return filteredData[filteredData.length - 2];
  };

  // Handle company selection
  const handleCompanyChange = (e) => {
    setSelectedCompany(e.target.value);
    setSelectedQuarters({ quarter1: null, quarter2: null });
  };

  // Handle quarter selection
  const handleQuarterChange = (quarterKey, e) => {
    const selectedDate = new Date(e.target.value);
    setSelectedQuarters(prev => ({
      ...prev,
      [quarterKey]: selectedDate
    }));
  };

  // Render KPI card
  const renderKpiCard = (title, value, change, isPositiveGood = true) => {
    const formattedValue = formatCurrency(value);
    const changeDirection = change > 0 ? 'up' : 'down';
    const changeClass = (changeDirection === 'up' && isPositiveGood) || (changeDirection === 'down' && !isPositiveGood) 
      ? 'text-green-600' 
      : 'text-red-600';
    
    return (
      <div className="bg-white p-4 rounded shadow">
        <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
        <p className="text-2xl font-bold">{formattedValue}</p>
        {change !== null && (
          <p className={`flex items-center ${changeClass}`}>
            {change.toFixed(2)}%
            <span className="ml-1">
              {changeDirection === 'up' ? '↑' : '↓'}
            </span>
          </p>
        )}
      </div>
    );
  };

  // Render ratio card
  const renderRatioCard = (title, value, description) => {
    return (
      <div className="bg-blue-50 p-4 rounded shadow">
        <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
        <p className="text-2xl font-bold">{value.toFixed(2)}%</p>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
    );
  };

  // Render a line chart
  const renderLineChart = (data, dataKey, title, color) => {
    return (
      <div className="bg-white p-4 rounded shadow h-64">
        <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
        <ResponsiveContainer width="100%" height="85%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="Quarter End Date" 
              tickFormatter={formatQuarterDate}
              tick={{ fontSize: 12 }}
            />
            <YAxis />
            <Tooltip 
              labelFormatter={formatQuarterDate}
              formatter={(value) => [formatCurrency(value), '']}
            />
            <Line 
              type="monotone" 
              dataKey={dataKey} 
              stroke={color} 
              strokeWidth={2}
              activeDot={{ r: 8 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Render expense comparison
  const renderExpenseComparison = () => {
    const q1Data = getQuarterData(selectedQuarters.quarter1);
    const q2Data = getQuarterData(selectedQuarters.quarter2);
    
    if (!q1Data || !q2Data) {
      return (
        <div className="bg-white p-4 rounded shadow">
          <p>Please select two quarters to compare expenses.</p>
        </div>
      );
    }
    
    const expenseCategories = [
      'Distribution Costs',
      'Administrative Expenses',
      'Other Expenses',
      'Finance Costs'
    ];
    
    // Filter out categories with no data
    const validCategories = expenseCategories.filter(
      category => q1Data[category] !== undefined && q2Data[category] !== undefined
    );
    
    if (validCategories.length === 0) {
      return (
        <div className="bg-white p-4 rounded shadow">
          <p>No expense data available for comparison.</p>
        </div>
      );
    }
    
    // Prepare data for bar chart
    const comparisonData = validCategories.map(category => ({
      name: category,
      [formatQuarterDate(selectedQuarters.quarter1)]: q1Data[category],
      [formatQuarterDate(selectedQuarters.quarter2)]: q2Data[category]
    }));
    
    // Prepare data for pie charts
    const q1PieData = validCategories.map(category => ({
      name: category,
      value: q1Data[category]
    }));
    
    const q2PieData = validCategories.map(category => ({
      name: category,
      value: q2Data[category]
    }));
    
    return (
      <div className="space-y-4">
        <div className="bg-white p-4 rounded shadow h-64">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Expense Comparison</h3>
          <ResponsiveContainer width="100%" height="85%">
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value) => [formatCurrency(value), '']} />
              <Legend />
              <Bar 
                dataKey={formatQuarterDate(selectedQuarters.quarter1)} 
                fill="#3498DB" 
              />
              <Bar 
                dataKey={formatQuarterDate(selectedQuarters.quarter2)} 
                fill="#E74C3C" 
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
        
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-white p-4 rounded shadow h-64">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              {formatQuarterDate(selectedQuarters.quarter1)} Distribution
            </h3>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie
                  data={q1PieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({name}) => name}
                >
                  {q1PieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [formatCurrency(value), '']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          
          <div className="bg-white p-4 rounded shadow h-64">
            <h3 className="text-lg font-semibold text-gray-700 mb-2">
              {formatQuarterDate(selectedQuarters.quarter2)} Distribution
            </h3>
            <ResponsiveContainer width="100%" height="85%">
              <PieChart>
                <Pie
                  data={q2PieData}
                  cx="50%"
                  cy="50%"
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  label={({name}) => name}
                >
                  {q2PieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [formatCurrency(value), '']} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-white p-4 rounded shadow">
          <h3 className="text-lg font-semibold text-gray-700 mb-2">Detailed Expense Comparison</h3>
          <table className="w-full">
            <thead>
              <tr className="bg-gray-100">
                <th className="p-2 text-left">Expense Category</th>
                <th className="p-2 text-right">{formatQuarterDate(selectedQuarters.quarter1)}</th>
                <th className="p-2 text-right">{formatQuarterDate(selectedQuarters.quarter2)}</th>
                <th className="p-2 text-right">Change</th>
              </tr>
            </thead>
            <tbody>
              {validCategories.map(category => {
                const q1Value = q1Data[category];
                const q2Value = q2Data[category];
                const change = calculateChange(q2Value, q1Value);
                const changeClass = change > 0 ? 'text-red-600' : 'text-green-600';
                
                return (
                  <tr key={category} className="border-b">
                    <td className="p-2">{category}</td>
                    <td className="p-2 text-right">{formatCurrency(q1Value)}</td>
                    <td className="p-2 text-right">{formatCurrency(q2Value)}</td>
                    <td className={`p-2 text-right ${changeClass}`}>
                      {change !== null ? `${change.toFixed(2)}% ${change > 0 ? '↑' : '↓'}` : 'N/A'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  // Main render
  if (loading) {
    return <div className="text-center p-8">Loading financial data...</div>;
  }

  if (error) {
    return <div className="text-center p-8 text-red-600">{error}</div>;
  }

  const filteredData = getFilteredData();
  const latestData = getLatestFinancials();
  const previousData = getPreviousFinancials();
  
  // Calculate changes if we have data
  const revenueChange = latestData && previousData ? 
    calculateChange(latestData['Revenue'], previousData['Revenue']) : null;
    
  const costChange = latestData && previousData ? 
    calculateChange(latestData['Cost of Sales'], previousData['Cost of Sales']) : null;
    
  const profitChange = latestData && previousData ? 
    calculateChange(latestData['Gross Profit'], previousData['Gross Profit']) : null;
    
  const operatingProfitChange = latestData && previousData ? 
    calculateChange(latestData['Profit from Operations'], previousData['Profit from Operations']) : null;

  // Calculate ratios
  const calculateRatios = () => {
    if (!latestData) return null;
    
    const grossMargin = (latestData['Gross Profit'] / latestData['Revenue']) * 100;
    const operatingMargin = (latestData['Profit from Operations'] / latestData['Revenue']) * 100;
    const distributionRatio = (latestData['Distribution Costs'] / latestData['Revenue']) * 100;
    
    return { grossMargin, operatingMargin, distributionRatio };
  };
  
  const ratios = calculateRatios();

  return (
    <div className="max-w-7xl mx-auto p-4 bg-gray-50">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Interactive Financial Dashboard</h1>
          <p className="text-gray-600">Analysis of quarterly financial performance</p>
        </div>
        <div className="text-sm text-gray-500">
          Last updated: {new Date().toLocaleString()}
        </div>
      </div>
      
      {/* Filters */}
      <div className="bg-white p-4 rounded shadow mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Select Company
            </label>
            <select 
              className="w-full p-2 border rounded"
              value={selectedCompany || ''}
              onChange={handleCompanyChange}
            >
              {companies.map(company => (
                <option key={company} value={company}>{company}</option>
              ))}
            </select>
          </div>
          
          <div className="flex flex-col">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Compare Quarters
            </label>
            <div className="grid grid-cols-2 gap-2">
              <select 
                className="w-full p-2 border rounded"
                value={selectedQuarters.quarter1 ? selectedQuarters.quarter1.toISOString() : ''}
                onChange={(e) => handleQuarterChange('quarter1', e)}
              >
                <option value="">Select First Quarter</option>
                {quarters.map(date => (
                  <option key={`q1-${date.getTime()}`} value={date.toISOString()}>
                    {formatQuarterDate(date)}
                  </option>
                ))}
              </select>
              
              <select 
                className="w-full p-2 border rounded"
                value={selectedQuarters.quarter2 ? selectedQuarters.quarter2.toISOString() : ''}
                onChange={(e) => handleQuarterChange('quarter2', e)}
              >
                <option value="">Select Second Quarter</option>
                {quarters.map(date => (
                  <option key={`q2-${date.getTime()}`} value={date.toISOString()}>
                    {formatQuarterDate(date)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>
      
      {latestData ? (
        <>
          {/* KPI Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            {renderKpiCard("Revenue", latestData['Revenue'], revenueChange)}
            {renderKpiCard("Cost of Sales", latestData['Cost of Sales'], costChange, false)}
            {renderKpiCard("Gross Profit", latestData['Gross Profit'], profitChange)}
            {renderKpiCard("Operating Profit", latestData['Profit from Operations'], operatingProfitChange)}
          </div>
          
          {/* Financial Trend Charts */}
          <h2 className="text-xl font-bold text-gray-800 mb-4">Financial Performance Trends</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {renderLineChart(filteredData, 'Revenue', 'Revenue Over Time', '#3498DB')}
            {renderLineChart(filteredData, 'Gross Profit', 'Gross Profit Over Time', '#2ECC71')}
            {renderLineChart(filteredData, 'Profit from Operations', 'Operating Profit Over Time', '#F1C40F')}
            {renderLineChart(filteredData, 'Profit for Period', 'Net Profit Over Time', '#9B59B6')}
          </div>
          
          {/* Financial Ratios */}
          {ratios && (
            <>
              <h2 className="text-xl font-bold text-gray-800 mb-4">Financial Ratios</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                {renderRatioCard("Gross Margin", ratios.grossMargin, "Gross Profit to Revenue")}
                {renderRatioCard("Operating Margin", ratios.operatingMargin, "Operating Profit to Revenue")}
                {renderRatioCard("Distribution to Revenue", ratios.distributionRatio, "Distribution efficiency")}
              </div>
            </>
          )}
          
          {/* Expense Analysis */}
          <h2 className="text-xl font-bold text-gray-800 mb-4">Expense Analysis</h2>
          {renderExpenseComparison()}
        </>
      ) : (
        <div className="text-center p-8">
          <p>No data available for the selected company.</p>
        </div>
      )}
      
      {/* Footer */}
      <div className="mt-8 text-center text-sm text-gray-500">
        Financial Dashboard © 2025
      </div>
    </div>
  );
};

export default FinancialDashboard;