import React, { useState, useEffect } from 'react';
import { BarChart3, Package, Users, Calendar, ShoppingCart, TrendingUp, Layout } from 'lucide-react';
import { initializeStorage } from './storage';
import POSSystem from './components/POSSystem';
import InventoryManager from './components/InventoryManager';
import SalesDashboard from './components/SalesDashboard';
import CustomerManager from './components/CustomerManager';
import AppointmentManager from './components/AppointmentManager';
import StoreDisplayPlanner from './components/StoreDisplayPlanner';
import './App.css';

type Tab = 'dashboard' | 'pos' | 'inventory' | 'customers' | 'appointments' | 'display';

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('dashboard');

  useEffect(() => {
    initializeStorage();
  }, []);

  const navItems = [
    { id: 'dashboard' as Tab, label: 'Dashboard', icon: TrendingUp },
    { id: 'pos' as Tab, label: 'POS', icon: ShoppingCart },
    { id: 'inventory' as Tab, label: 'Inventory', icon: Package },
    { id: 'customers' as Tab, label: 'Customers', icon: Users },
    { id: 'appointments' as Tab, label: 'Appointments', icon: Calendar },
    { id: 'display' as Tab, label: 'Store Layout', icon: Layout },
  ];

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <BarChart3 size={28} />
            <h1>Medical Store & Clinic Manager</h1>
          </div>
          <p className="tagline">Patanjali Store Management System</p>
        </div>
      </header>

      <nav className="app-nav">
        {navItems.map(item => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? 'active' : ''}`}
              onClick={() => setActiveTab(item.id)}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <main className="app-main">
        {activeTab === 'dashboard' && <SalesDashboard />}
        {activeTab === 'pos' && <POSSystem />}
        {activeTab === 'inventory' && <InventoryManager />}
        {activeTab === 'customers' && <CustomerManager />}
        {activeTab === 'appointments' && <AppointmentManager />}
        {activeTab === 'display' && <StoreDisplayPlanner />}
      </main>
    </div>
  );
}

export default App;
