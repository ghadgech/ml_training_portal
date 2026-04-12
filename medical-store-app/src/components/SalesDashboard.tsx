import React, { useMemo } from 'react';
import { TrendingUp, TrendingDown, DollarSign, ShoppingCart } from 'lucide-react';
import { getBills, getProducts, getAppointments } from '../storage';
import '../styles/SalesDashboard.css';

export default function SalesDashboard() {
  const bills = getBills();
  const products = getProducts();
  const appointments = getAppointments();

  const stats = useMemo(() => {
    const today = new Date().toDateString();
    const thisMonth = new Date().getMonth();
    const thisYear = new Date().getFullYear();

    // Today's sales
    const todayBills = bills.filter(b => new Date(b.date).toDateString() === today);
    const todayRevenue = todayBills.reduce((sum, b) => sum + b.totalAmount, 0);
    const todayProfit = todayBills.reduce((profit, bill) => {
      bill.items.forEach(item => {
        const product = products.find(p => p.id === item.productId);
        if (product) {
          profit += (product.sellingPrice - product.cost) * item.quantity;
        }
      });
      return profit;
    }, 0);

    // This month
    const monthBills = bills.filter(b => {
      const date = new Date(b.date);
      return date.getMonth() === thisMonth && date.getFullYear() === thisYear;
    });
    const monthRevenue = monthBills.reduce((sum, b) => sum + b.totalAmount, 0);
    const monthProfit = monthBills.reduce((profit, bill) => {
      bill.items.forEach(item => {
        const product = products.find(p => p.id === item.productId);
        if (product) {
          profit += (product.sellingPrice - product.cost) * item.quantity;
        }
      });
      return profit;
    }, 0);

    // Overall
    const totalRevenue = bills.reduce((sum, b) => sum + b.totalAmount, 0);
    const totalProfit = bills.reduce((profit, bill) => {
      bill.items.forEach(item => {
        const product = products.find(p => p.id === item.productId);
        if (product) {
          profit += (product.sellingPrice - product.cost) * item.quantity;
        }
      });
      return profit;
    }, 0);

    // Payment methods
    const paymentBreakdown = {
      cash: bills.filter(b => b.paymentMethod === 'cash').reduce((sum, b) => sum + b.totalAmount, 0),
      card: bills.filter(b => b.paymentMethod === 'card').reduce((sum, b) => sum + b.totalAmount, 0),
      upi: bills.filter(b => b.paymentMethod === 'upi').reduce((sum, b) => sum + b.totalAmount, 0),
    };

    // Top products
    const productSales: { [key: string]: { quantity: number; revenue: number } } = {};
    bills.forEach(bill => {
      bill.items.forEach(item => {
        if (!productSales[item.productId]) {
          productSales[item.productId] = { quantity: 0, revenue: 0 };
        }
        productSales[item.productId].quantity += item.quantity;
        productSales[item.productId].revenue += item.price * item.quantity;
      });
    });

    const topProducts = Object.entries(productSales)
      .map(([id, data]) => ({
        id,
        name: products.find(p => p.id === id)?.name || 'Unknown',
        ...data,
      }))
      .sort((a, b) => b.revenue - a.revenue)
      .slice(0, 5);

    // Appointments
    const completedAppointments = appointments.filter(a => a.status === 'completed').length;
    const scheduledAppointments = appointments.filter(a => a.status === 'scheduled').length;

    return {
      todayRevenue,
      todayProfit,
      monthRevenue,
      monthProfit,
      totalRevenue,
      totalProfit,
      paymentBreakdown,
      topProducts,
      totalBills: bills.length,
      completedAppointments,
      scheduledAppointments,
      inventoryValue: products.reduce((sum, p) => sum + p.cost * p.stock, 0),
    };
  }, [bills, products, appointments]);

  return (
    <div className="sales-dashboard">
      <h2 className="card-title">Dashboard Overview</h2>

      {/* Key Metrics */}
      <div className="grid-4">
        <div className="stat-card">
          <div className="stat-label">Today's Revenue</div>
          <div className="stat-value">₹{stats.todayRevenue.toLocaleString()}</div>
          <div className="stat-change">+12% from yesterday</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Today's Profit</div>
          <div className="stat-value">₹{stats.todayProfit.toFixed(0)}</div>
          <div className="stat-change profit-text">
            Margin: {stats.todayRevenue > 0 ? ((stats.todayProfit / stats.todayRevenue) * 100).toFixed(1) : 0}%
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Transactions</div>
          <div className="stat-value">{stats.totalBills}</div>
          <div className="stat-change">All time</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Inventory Value</div>
          <div className="stat-value">₹{stats.inventoryValue.toLocaleString()}</div>
          <div className="stat-change">{products.length} products</div>
        </div>
      </div>

      <div className="grid-2">
        {/* Monthly Performance */}
        <div className="card">
          <h3 className="card-title">Monthly Performance</h3>
          <div className="metric-box">
            <div className="metric">
              <div className="metric-label">Revenue</div>
              <div className="metric-value">₹{stats.monthRevenue.toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Profit</div>
              <div className="metric-value profit">₹{stats.monthProfit.toFixed(0)}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Transactions</div>
              <div className="metric-value">{bills.filter(b => {
                const date = new Date(b.date);
                const now = new Date();
                return date.getMonth() === now.getMonth() && date.getFullYear() === now.getFullYear();
              }).length}</div>
            </div>
          </div>
        </div>

        {/* Payment Methods */}
        <div className="card">
          <h3 className="card-title">Payment Methods</h3>
          <div className="metric-box">
            <div className="metric">
              <div className="metric-label">Cash</div>
              <div className="metric-value">₹{stats.paymentBreakdown.cash.toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Card</div>
              <div className="metric-value">₹{stats.paymentBreakdown.card.toLocaleString()}</div>
            </div>
            <div className="metric">
              <div className="metric-label">UPI</div>
              <div className="metric-value">₹{stats.paymentBreakdown.upi.toLocaleString()}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        {/* Top Selling Products */}
        <div className="card">
          <h3 className="card-title">Top Selling Products</h3>
          {stats.topProducts.length === 0 ? (
            <p className="empty-state">No sales yet</p>
          ) : (
            <div className="product-list">
              {stats.topProducts.map((product, index) => (
                <div key={product.id} className="product-item">
                  <div className="product-rank">#{index + 1}</div>
                  <div className="product-details">
                    <div className="product-name">{product.name}</div>
                    <div className="product-meta">{product.quantity} units sold</div>
                  </div>
                  <div className="product-revenue">₹{product.revenue.toLocaleString()}</div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Clinic Appointments */}
        <div className="card">
          <h3 className="card-title">Clinic Status</h3>
          <div className="metric-box">
            <div className="metric">
              <div className="metric-label">Scheduled</div>
              <div className="metric-value">{stats.scheduledAppointments}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Completed</div>
              <div className="metric-value">{stats.completedAppointments}</div>
            </div>
            <div className="metric">
              <div className="metric-label">Pending</div>
              <div className="metric-value">{stats.scheduledAppointments}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Overall Metrics */}
      <div className="card">
        <h3 className="card-title">All-Time Metrics</h3>
        <div className="metric-box wide">
          <div className="metric">
            <div className="metric-label">Total Revenue</div>
            <div className="metric-value large">₹{stats.totalRevenue.toLocaleString()}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Total Profit</div>
            <div className="metric-value large profit">₹{stats.totalProfit.toFixed(0)}</div>
          </div>
          <div className="metric">
            <div className="metric-label">Average Profit Margin</div>
            <div className="metric-value large">
              {stats.totalRevenue > 0 ? ((stats.totalProfit / stats.totalRevenue) * 100).toFixed(1) : 0}%
            </div>
          </div>
          <div className="metric">
            <div className="metric-label">Average Transaction</div>
            <div className="metric-value large">
              ₹{stats.totalBills > 0 ? (stats.totalRevenue / stats.totalBills).toFixed(0) : 0}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
